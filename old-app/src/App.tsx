import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import "./App.css";
import { CSVDataLoader, FlightDataRow } from "./lib/data-loader";
import LineChart from "./components/line-chart";

const loader = new CSVDataLoader();


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
                {dataQuery.data.length} rows found.
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="accel_x" width={1000} height={400} />
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="accel_y" width={1000} height={400} />
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="accel_z" width={1000} height={400} />
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="gyro_x" width={1000} height={400} />
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="gyro_y" width={1000} height={400} />
                <LineChart<FlightDataRow> series={dataQuery.data} timeKey="timestamp" valueKey="gyro_z" width={1000} height={400} />
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
