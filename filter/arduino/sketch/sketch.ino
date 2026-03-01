#include <Wire.h>

// --- Protocol ---
// Packet: [0xAA] [type:u8] [len:u8] [timestamp_us:u32 LE] [payload: len bytes] [crc8]
// crc8 = XOR of all bytes after sync (type, len, timestamp, payload)
constexpr uint8_t SYNC = 0xAA;
constexpr uint8_t PKT_IMU  = 0x01; // payload: accel[3] + mag[3] + gyro[3], all int16 LE (18B)
constexpr uint8_t PKT_BARO = 0x02; // payload: pressure int32 LE, centipascals (4B)

// --- Sample rates ---
constexpr uint32_t IMU_INTERVAL_US  = 10000; // 100 Hz
constexpr uint32_t BARO_INTERVAL_US = 40000; //  25 Hz

// --- BNO055 ---
constexpr uint8_t BNO_ADDR         = 0x28;
constexpr uint8_t BNO_CHIP_ID      = 0xA0;
constexpr uint8_t BNO_REG_CHIP_ID  = 0x00;
constexpr uint8_t BNO_REG_OPR_MODE = 0x3D;
constexpr uint8_t BNO_REG_PWR_MODE = 0x3E;
constexpr uint8_t BNO_REG_SYS_TRIGGER = 0x3F;
constexpr uint8_t BNO_REG_UNIT_SEL = 0x3B;
constexpr uint8_t BNO_REG_ACC_DATA = 0x08; // 18 contiguous bytes: accel(6) + mag(6) + gyro(6)
constexpr uint8_t BNO_MODE_AMG     = 0x07; // raw accel + mag + gyro, no fusion

// --- BMP280 ---
constexpr uint8_t BMP_ADDR            = 0x77;
constexpr uint8_t BMP_CHIP_ID         = 0x58;
constexpr uint8_t BMP_REG_CHIP_ID     = 0xD0;
constexpr uint8_t BMP_REG_RESET       = 0xE0;
constexpr uint8_t BMP_REG_CTRL_MEAS   = 0xF4;
constexpr uint8_t BMP_REG_CONFIG      = 0xF5;
constexpr uint8_t BMP_REG_PRESS_MSB   = 0xF7; // 6 contiguous bytes: press(3) + temp(3)
constexpr uint8_t BMP_REG_CALIB       = 0x88; // 26 bytes of calibration data

// --- BMP280 calibration coefficients ---
struct Bmp280Calib {
  uint16_t dig_T1;
  int16_t  dig_T2, dig_T3;
  uint16_t dig_P1;
  int16_t  dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9;
};

Bmp280Calib bmp_calib;
int32_t bmp_t_fine;
uint32_t next_imu_us;
uint32_t next_baro_us;

// --- I2C helpers ---

void write_reg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t read_reg(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.read();
}

void read_bytes(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, len);
  for (uint8_t i = 0; i < len; i++) {
    buf[i] = Wire.read();
  }
}

// --- Protocol ---

void send_packet(uint8_t type, uint32_t timestamp_us, const uint8_t* payload, uint8_t len) {
  uint8_t header[7];
  header[0] = SYNC;
  header[1] = type;
  header[2] = len;
  memcpy(&header[3], &timestamp_us, 4);

  uint8_t crc = 0;
  for (uint8_t i = 1; i < 7; i++) crc ^= header[i];
  for (uint8_t i = 0; i < len; i++) crc ^= payload[i];

  Serial.write(header, 7);
  Serial.write(payload, len);
  Serial.write(crc);
}

// --- BNO055 ---

void bno055_init() {
  uint8_t id = read_reg(BNO_ADDR, BNO_REG_CHIP_ID);
  if (id != BNO_CHIP_ID) {
    Serial.print("ERROR: BNO055 chip ID 0x");
    Serial.println(id, HEX);
    while (true) {}
  }

  write_reg(BNO_ADDR, BNO_REG_SYS_TRIGGER, 0x20); // reset
  delay(700);
  write_reg(BNO_ADDR, BNO_REG_PWR_MODE, 0x00); // normal power
  delay(10);
  write_reg(BNO_ADDR, BNO_REG_UNIT_SEL, 0x02); // m/s², rad/s
  write_reg(BNO_ADDR, BNO_REG_OPR_MODE, BNO_MODE_AMG);
  delay(20);
}

// --- BMP280 ---

void bmp280_init() {
  uint8_t id = read_reg(BMP_ADDR, BMP_REG_CHIP_ID);
  if (id != BMP_CHIP_ID) {
    Serial.print("ERROR: BMP280 chip ID 0x");
    Serial.println(id, HEX);
    while (true) {}
  }

  write_reg(BMP_ADDR, BMP_REG_RESET, 0xB6);
  delay(100);

  // Read calibration (26 bytes, struct layout matches register order)
  uint8_t calib_raw[26];
  read_bytes(BMP_ADDR, BMP_REG_CALIB, calib_raw, 26);
  memcpy(&bmp_calib, calib_raw, 26);

  // Config: standby 0.5ms, IIR filter coeff 16
  write_reg(BMP_ADDR, BMP_REG_CONFIG, (0x00 << 5) | (0x04 << 2));
  // ctrl_meas: temp oversampling x2, pressure oversampling x16, normal mode
  write_reg(BMP_ADDR, BMP_REG_CTRL_MEAS, (0x02 << 5) | (0x05 << 2) | 0x03);
}

int32_t bmp280_read_pressure() {
  uint8_t raw[6];
  read_bytes(BMP_ADDR, BMP_REG_PRESS_MSB, raw, 6);

  int32_t adc_P = ((int32_t)raw[0] << 12) | ((int32_t)raw[1] << 4) | (raw[2] >> 4);
  int32_t adc_T = ((int32_t)raw[3] << 12) | ((int32_t)raw[4] << 4) | (raw[5] >> 4);

  // Temperature compensation (needed internally for pressure calc)
  int32_t var1 = ((((adc_T >> 3) - ((int32_t)bmp_calib.dig_T1 << 1))) *
                  (int32_t)bmp_calib.dig_T2) >> 11;
  int32_t var2 = (((((adc_T >> 4) - (int32_t)bmp_calib.dig_T1) *
                    ((adc_T >> 4) - (int32_t)bmp_calib.dig_T1)) >> 12) *
                  (int32_t)bmp_calib.dig_T3) >> 14;
  bmp_t_fine = var1 + var2;

  // Pressure compensation (BMP280 datasheet algorithm, returns Pa in Q24.8)
  int64_t v1 = (int64_t)bmp_t_fine - 128000;
  int64_t v2 = v1 * v1 * (int64_t)bmp_calib.dig_P6;
  v2 = v2 + ((v1 * (int64_t)bmp_calib.dig_P5) << 17);
  v2 = v2 + ((int64_t)bmp_calib.dig_P4 << 35);
  v1 = ((v1 * v1 * (int64_t)bmp_calib.dig_P3) >> 8) +
       ((v1 * (int64_t)bmp_calib.dig_P2) << 12);
  v1 = (((int64_t)1 << 47) + v1) * (int64_t)bmp_calib.dig_P1 >> 33;
  if (v1 == 0) return 0;

  int64_t p = 1048576 - adc_P;
  p = (((p << 31) - v2) * 3125) / v1;
  v1 = ((int64_t)bmp_calib.dig_P9 * (p >> 13) * (p >> 13)) >> 25;
  v2 = ((int64_t)bmp_calib.dig_P8 * p) >> 19;
  p = ((p + v1 + v2) >> 8) + ((int64_t)bmp_calib.dig_P7 << 4);

  // Convert from Pa*256 (Q24.8) to centipascals (Pa*100)
  return (int32_t)((p * 100) / 256);
}

// --- Main ---

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  bno055_init();
  bmp280_init();

  next_imu_us = micros();
  next_baro_us = next_imu_us;
}

void loop() {
  uint32_t now = micros();

  if ((int32_t)(now - next_imu_us) >= 0) {
    next_imu_us += IMU_INTERVAL_US;
    uint8_t raw[18];
    read_bytes(BNO_ADDR, BNO_REG_ACC_DATA, raw, 18);
    send_packet(PKT_IMU, now, raw, 18);
  }

  if ((int32_t)(now - next_baro_us) >= 0) {
    next_baro_us += BARO_INTERVAL_US;
    int32_t pressure = bmp280_read_pressure();
    send_packet(PKT_BARO, now, (const uint8_t*)&pressure, 4);
  }
}
