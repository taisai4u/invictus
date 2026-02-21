import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import "./App.css";
import { CSVDataLoader, FlightDataRow } from "./lib/data-loader";

const loader = new CSVDataLoader();

const COLUMNS: { key: keyof FlightDataRow; label: string }[] = [
  { key: "timestamp", label: "Time (s)" },
  { key: "accel_x", label: "Accel X" },
  { key: "accel_y", label: "Accel Y" },
  { key: "accel_z", label: "Accel Z" },
  { key: "gyro_x", label: "Gyro X" },
  { key: "gyro_y", label: "Gyro Y" },
  { key: "gyro_z", label: "Gyro Z" },
  { key: "mag_x", label: "Mag X" },
  { key: "mag_y", label: "Mag Y" },
  { key: "mag_z", label: "Mag Z" },
  { key: "pressure", label: "Pressure" },
  { key: "temperature", label: "Temp (°C)" },
];

function fmt(val: number) {
  return typeof val === "number" ? val.toFixed(4) : val;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const fileListQuery = useQuery({
    queryKey: ["files"],
    queryFn: () => loader.listFiles(),
  });

  const dataQuery = useQuery({
    queryKey: ["data", selectedFile],
    queryFn: () => loader.loadData(selectedFile!),
    enabled: selectedFile !== null,
  });

  return (
    <div>
      <h1>Invictus</h1>
      {/* Sidebar */}
      <aside>
        <h2>Data Files</h2>

        {fileListQuery.isLoading && <p className="text-muted-foreground">Loading files…</p>}

        {fileListQuery.isError && (
          <p className="text-destructive">{String(fileListQuery.error)}</p>
        )}

        {fileListQuery.isSuccess && fileListQuery.data.length === 0 && (
          <p className="text-muted-foreground">No files found.</p>
        )}

        <ul>
          {fileListQuery.data?.map((f) => (
            <li key={f} className={selectedFile === f ? "text-primary" : ""}>
              <span>{f}</span>
              <button onClick={() => setSelectedFile(f)}>
                Load
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* Main content */}
      <main>
        {!selectedFile && (
          <div>
            <p>Select a file from the sidebar to view its data.</p>
          </div>
        )}

        {selectedFile && (
          <>
            <h2>{selectedFile}</h2>

            {dataQuery.isLoading && <p className="text-muted-foreground">Loading…</p>}

            {dataQuery.isError && (
              <p className="text-destructive">{String(dataQuery.error)}</p>
            )}

            {dataQuery.isSuccess && dataQuery.data.length > 0 && (
              <div>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      {COLUMNS.map((c) => (
                        <th key={c.key}>{c.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {dataQuery.data.map((row, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        {COLUMNS.map((c) => (
                          <td key={c.key}>{fmt(row[c.key] as number)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {dataQuery.isSuccess && dataQuery.data.length === 0 && (
              <p className="text-muted-foreground">No data rows found.</p>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
