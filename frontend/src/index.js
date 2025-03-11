
import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
// Polyfill for Node.js core modules required by simple-peer
import { Buffer } from "buffer";
import stream from "stream-browserify";

window.Buffer = Buffer; // Polyfill for Buffer
window.stream = stream; // Polyfill for stream

// Define a global process object to prevent the error
window.process = {
  env: {
    NODE_ENV: "development", // or "production"
  },
  nextTick: (fn) => {
    setTimeout(fn, 0); // Polyfill for process.nextTick
  },
};

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
);

reportWebVitals();
