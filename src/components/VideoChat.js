import React, { useRef, useState } from "react";
import SimplePeer from "simple-peer";
import "./Video.css";

const VideoChat = () => {
  const [myStream, setMyStream] = useState(null);
  const [peerStream, setPeerStream] = useState(null);
  const [connection, setConnection] = useState(null);

  const myVideoRef = useRef();
  const peerVideoRef = useRef();
  const signalInputRef = useRef();

  // Start local video stream
  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      setMyStream(stream);
      myVideoRef.current.srcObject = stream;
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  };

  // Create a new peer connection
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
        </div>
        <div>
          <h3>Peer Video</h3>
          <video ref={peerVideoRef} autoPlay></video>
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
    </div>
  );
};

export default VideoChat;
