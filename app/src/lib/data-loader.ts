import Papa from "papaparse";

import { BaseDirectory, readDir, readTextFile } from '@tauri-apps/plugin-fs';

const DATA_DIR = "dev/invictus/tools/analyzer/assets/sample_data";

// barometer records: barometric pressure, temperature, humidity
export interface FlightDataRow {
  timestamp: number;
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  mag_x: number;
  mag_y: number;
  mag_z: number;
  pressure: number;
  temperature: number;
}

export interface DataLoader {
  listFiles(): Promise<string[]>;
  loadData(filename: string): Promise<FlightDataRow[]>;
}

export class CSVDataLoader implements DataLoader {
  async listFiles(): Promise<string[]> {
    const files = await readDir(DATA_DIR, { baseDir: BaseDirectory.Home });
    return files.map((file) => file.name);
  }

  async loadData(filename: string) {
    const filepath = `${DATA_DIR}/${filename}`;
    const data = await readTextFile(filepath, { baseDir: BaseDirectory.Home });
    const result = Papa.parse<FlightDataRow>(data, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });
    return result.data;
  }
}