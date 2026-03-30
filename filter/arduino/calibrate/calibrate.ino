#include <Adafruit_BNO055.h>
#include <Wire.h>

Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28, &Wire);

void displayCalStatus(void)
{
  uint8_t system, gyro, accel, mag;
  system = gyro = accel = mag = 0;
  bno.getCalibration(&system, &gyro, &accel, &mag);

  Serial.print("\t");
  if (!system)
  {
    Serial.print("! ");
  }

  Serial.print("Sys:");
  Serial.print(system, DEC);
  Serial.print(" G:");
  Serial.print(gyro, DEC);
  Serial.print(" A:");
  Serial.print(accel, DEC);
  Serial.print(" M:");
  Serial.println(mag, DEC);
}

void displaySensorOffsets(void)
{
  adafruit_bno055_offsets_t offsets;
  bno.getSensorOffsets(offsets);

  Serial.println("Sensor offsets:");
  Serial.print("  Accel: X="); Serial.print(offsets.accel_offset_x);
  Serial.print(" Y=");         Serial.print(offsets.accel_offset_y);
  Serial.print(" Z=");         Serial.println(offsets.accel_offset_z);
  Serial.print("  Gyro:  X="); Serial.print(offsets.gyro_offset_x);
  Serial.print(" Y=");         Serial.print(offsets.gyro_offset_y);
  Serial.print(" Z=");         Serial.println(offsets.gyro_offset_z);
  Serial.print("  Mag:   X="); Serial.print(offsets.mag_offset_x);
  Serial.print(" Y=");         Serial.print(offsets.mag_offset_y);
  Serial.print(" Z=");         Serial.println(offsets.mag_offset_z);
  Serial.print("  Accel radius="); Serial.println(offsets.accel_radius);
  Serial.print("  Mag radius=");   Serial.println(offsets.mag_radius);
}

void setup(){
  Serial.begin(9600);
  Wire.begin();
  Wire.setClock(400000);
  bno.begin();
}

void loop(){
  displayCalStatus();

  uint8_t system, gyro, accel, mag;
  system = gyro = accel = mag = 0;
  bno.getCalibration(&system, &gyro, &accel, &mag);

  if (system == 3 && gyro == 3 && accel == 3 && mag == 3)
  {
    Serial.println("\nFully calibrated!");
    displaySensorOffsets();
    delay(5000);
  }
}