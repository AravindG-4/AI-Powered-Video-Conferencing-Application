import React, { useRef, useState } from "react";
import SimplePeer from "simple-peer";
import "./Video.css";

const VideoChat = () => {
  const [myStream, setMyStream] = useState(null);
  const [peerStream, setPeerStream] = useState(null);
  const [connection, setConnection] = useState(null);
  const [ws, setWs] = useState(null); // WebSocket state
  const [myEmotion, setMyEmotion] = useState("neutral"); // State to hold my detected emotion
  const [peerEmotion, setPeerEmotion] = useState("neutral"); // State to hold peer's detected emotion
  const myVideoRef = useRef();
  const peerVideoRef = useRef();
  const signalInputRef = useRef();

  // Start local video stream with optimized constraints
  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      setMyStream(stream);
      myVideoRef.current.srcObject = stream;

      // Start sending frames to the API
      startSendingFrames(stream);
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  };

  // Function to start sending frames to the API
  const startSendingFrames = (stream) => {
    const videoTrack = stream.getVideoTracks()[0];
    const imageCapture = new ImageCapture(videoTrack);

    // Create WebSocket connection
    const websocket = new WebSocket("ws://10.1.58.223:8000/ws/emotion-detection"); // Change to wss if needed
    setWs(websocket);

    websocket.onopen = () => {
      console.log("WebSocket connection established");
    };

    websocket.onmessage = (event) => {
      console.log("Received emotion:", event.data);
      setPeerEmotion(event.data); // Update peer's emotion state
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    websocket.onclose = () => {
      console.log("WebSocket connection closed");
      // Optionally attempt to reconnect here if needed
    };

    // Capture frames at a regular interval
    const captureFrame = async () => {
      try {
        const bitmap = await imageCapture.grabFrame();
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(bitmap, 0, 0);

        // Convert canvas to base64 image
        const base64ImageData = canvas.toDataURL("image/jpeg");

        // Send base64 image data to the WebSocket server if connected
        if (websocket.readyState === WebSocket.OPEN) {
          websocket.send(base64ImageData.split(",")[1]); // Send only the base64 part
        }
      } catch (error) {
        console.error("Error capturing frame:", error);
      }
    };

    // Capture a frame every second (adjust as needed)
    const frameInterval = setInterval(captureFrame, 1000);

    websocket.onclose = () => {
      clearInterval(frameInterval); // Clear interval on WebSocket close
      console.log("WebSocket connection closed");
    };
  };

  // Create a new peer connection with optimized settings
  const createConnection = () => {
    const peer = new SimplePeer({
      initiator: true,
      trickle: false,
      stream: myStream,
    });

    peer.on("signal", (data) => {
      console.log("Signal data:", JSON.stringify(data));
      alert("Share this signal with your peer:\n" + JSON.stringify(data));
    });

    peer.on("stream", (stream) => {
      setPeerStream(stream);
      peerVideoRef.current.srcObject = stream;
    });

    setConnection(peer);
  };

  // Join an existing connection using a signal
  const joinConnection = () => {
    if (!connection) {
      const peer = new SimplePeer({
        initiator: false,
        trickle: false,
        stream: myStream,
      });

      peer.on("signal", (data) => {
        console.log("Signal data:", JSON.stringify(data));
        alert("Share this signal with your peer:\n" + JSON.stringify(data));
      });

      peer.on("stream", (stream) => {
        setPeerStream(stream);
        peerVideoRef.current.srcObject = stream;
      });

      setConnection(peer);
    }

    const signalData = JSON.parse(signalInputRef.current.value);
    connection.signal(signalData);
  };

  return (
    <div className="video-chat">
      <div className="video-container">
        <div>
          <h3>My Video</h3>
          <video ref={myVideoRef} autoPlay muted></video>
          <div className="emotion-display">My Emotion: {peerEmotion}</div> {/* Display my emotion here */}
        </div>
        <div>
          <h3>Peer Video</h3>
          <video ref={peerVideoRef} autoPlay></video>
          <div className="emotion-display">Peer's Emotion: {myEmotion}</div> {/* Display peer's emotion here */}
        </div>
      </div>

      <div className="controls">
        {!myStream && <button onClick={startStream}>Start My Stream</button>}
        {myStream && !connection && (
          <button onClick={createConnection}>Create Connection</button>
        )}
        {myStream && connection && (
          <>
            <textarea ref={signalInputRef} placeholder="Paste signal here"></textarea>
            <button onClick={joinConnection}>Join Connection</button>
          </>
        )}
      </div>

      {/* Emotion display styling */}
      <style jsx>{`
        .emotion-display {
          font-size: 24px;
          font-weight: bold;
          color: #333;
          margin-top: 10px;
          text-align: center;
        }
      `}</style>
      
    </div>
  );
};

export default VideoChat;
