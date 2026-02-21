import { useState } from "react";
import reactLogo from "./assets/react.svg";
// import { invoke } from "@tauri-apps/api/core";
import "./App.css";

import { BaseDirectory, readTextFile } from '@tauri-apps/plugin-fs';

function App() {
  const [filePath, setFilePath] = useState("");
  const [fileContent, setFileContent] = useState("");

  async function readFile() {
    // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
    // setGreetMsg(await invoke("greet", { name }));
    const content = await readTextFile(filePath, { baseDir: BaseDirectory.Home });
    setFileContent(content);
  }

  return (
    <main className="container">
      <h1>Welcome to Tauri + React</h1>

      <div className="row">
        <a href="https://vite.dev" target="_blank">
          <img src="/vite.svg" className="logo vite" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank">
          <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <p>Read a file</p>

      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          readFile();
        }}
      >
        <input
          id="greet-input"
          onChange={(e) => setFilePath(e.currentTarget.value)}
          placeholder="Enter a file path..."
        />
        <button type="submit">Greet</button>
      </form>
      <pre className="text-left">{fileContent}</pre>
    </main>
  );
}

export default App;
