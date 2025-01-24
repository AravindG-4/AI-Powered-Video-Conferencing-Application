const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
  },
});

const clients = {};

io.on("connection", (socket) => {
  console.log("New client connected:", socket.id);

  socket.on("sendEmotion", (data) => {
    console.log(`Received emotion: ${data.emotion} from peerId: ${data.fromPeerId}`);
    
    // Broadcast to all other clients except the sender
    socket.broadcast.emit("receiveEmotion", {
      emotion: data.emotion,
      fromPeerId: data.fromPeerId
    });
  });

  socket.on("disconnect", () => {
    console.log(`Client disconnected: ${socket.id}`);
  });
});

server.listen(5000, () => {
  console.log("Socket.IO server running on http://10.1.58.223:5000");
});