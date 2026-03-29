#include <Adafruit_BMP280.h>
#include <Adafruit_BNO055.h>
#include <Wire.h>
#include <utility/imumaths.h>

// --- Protocol ---
// Packet: [0xAA] [type:u8] [len:u8] [timestamp_us:u32 LE] [payload: len bytes]
// [crc8] crc8 = XOR of all bytes after sync (type, len, timestamp, payload)
constexpr uint8_t SYNC = 0xAA;
constexpr uint8_t PKT_IMU =
    0x01; // payload: accel[3] + mag[3] + gyro[3], all int16 LE (18B)
constexpr uint8_t PKT_BARO =
    0x02; // payload: pressure int32 LE, centipascals (4B)

// --- Sample rates ---
constexpr uint32_t IMU_INTERVAL_US = 10000;   // 100 Hz
constexpr uint32_t BARO_INTERVAL_US = 250000; //  4 Hz

// Check I2C device address and correct line below (by default address is 0x29
// or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28, &Wire);
Adafruit_BMP280 bmp; // I2C

uint32_t next_imu_us;
uint32_t next_baro_us;
uint32_t time_origin_us;

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

void read_bytes(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, len);
  for (uint8_t i = 0; i < len; i++) {
    buf[i] = Wire.read();
  }
}

// --- Protocol ---

void send_packet(uint8_t type, uint32_t timestamp_us, const uint8_t *payload,
                 uint8_t len) {
  uint8_t header[7];
  header[0] = SYNC;
  header[1] = type;
  header[2] = len;
  memcpy(&header[3], &timestamp_us, 4);

  uint8_t crc = 0;
  for (uint8_t i = 1; i < 7; i++)
    crc ^= header[i];
  for (uint8_t i = 0; i < len; i++)
    crc ^= payload[i];

  Serial.write(header, 7);
  Serial.write(payload, len);
  Serial.write(crc);
}

// --- BNO055 ---

void bno055_init() {
  if (!bno.begin()) {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print(
        "Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1)
      ;
  }
}

// --- BMP280 ---

void bmp280_init() {
  if (!bmp.begin()) {
    Serial.println(
        "ERROR: Could not find a valid BMP280 sensor, check wiring!");
    Serial.print("SensorID was: 0x");
    Serial.println(bmp.sensorID(), 16);
    Serial.print("        ID of 0xFF probably means a bad address, a BMP 180 "
                 "or BMP 085\n");
    Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
    Serial.print("        ID of 0x60 represents a BME 280.\n");
    Serial.print("        ID of 0x61 represents a BME 680.\n");
    while (1)
      delay(10);
  }
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                  Adafruit_BMP280::SAMPLING_X1, // temperature
                  Adafruit_BMP280::SAMPLING_X1, // pressure
                  Adafruit_BMP280::FILTER_OFF,  // filtering
                  Adafruit_BMP280::STANDBY_MS_250);
}

// --- Main ---

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  bno055_init();
  bmp280_init();

  time_origin_us = micros();
  next_imu_us = time_origin_us;
  next_baro_us = time_origin_us;
}

void loop() {
  uint32_t now = micros();

  if ((int32_t)(now - next_imu_us) >= 0) {
    next_imu_us += IMU_INTERVAL_US;
    imu::Vector<3> accel =
        bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // m/s^2
    imu::Vector<3> gyro =
        bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE); // deg/s
    imu::Vector<3> mag =
        bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER); // μT
    // Convert back to raw int16 values
    int16_t raw[9] = {
        (int16_t)(accel.x() * 100), (int16_t)(accel.y() * 100),
        (int16_t)(accel.z() * 100), (int16_t)(mag.x() * 16),
        (int16_t)(mag.y() * 16),    (int16_t)(mag.z() * 16),
        (int16_t)(gyro.x() * 16),   (int16_t)(gyro.y() * 16),
        (int16_t)(gyro.z() * 16),
    };
    send_packet(PKT_IMU, now - time_origin_us, (const uint8_t *)raw, 18);
  }

  if ((int32_t)(now - next_baro_us) >= 0) {
    next_baro_us += BARO_INTERVAL_US;
    int32_t pressure = bmp.readPressure();
    send_packet(PKT_BARO, now - time_origin_us, (const uint8_t *)&pressure, 4);
  }
}
