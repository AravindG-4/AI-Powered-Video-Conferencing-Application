let localStream;
let remoteStream;
let peerConnection;

const servers = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" } // Public STUN server
  ]
};

const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const startCallButton = document.getElementById("startCall");
const createOfferButton = document.getElementById("createOffer");
const setAnswerButton = document.getElementById("setAnswer");
const offerTextarea = document.getElementById("offer");
const answerTextarea = document.getElementById("answer");

let iceCandidates = [];

// Get access to local media
startCallButton.addEventListener("click", async () => {
  localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  remoteStream = new MediaStream();

  localVideo.srcObject = localStream;
  remoteVideo.srcObject = remoteStream;

  // Initialize PeerConnection
  peerConnection = new RTCPeerConnection(servers);

  // Add local stream tracks to the peer connection
  localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));

  // Listen for remote stream tracks
  peerConnection.ontrack = (event) => {
    event.streams[0].getTracks().forEach(track => remoteStream.addTrack(track));
  };

  // ICE Candidate handling
  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      iceCandidates.push(event.candidate);
      console.log("New ICE candidate:", event.candidate);
    }
  };
});

// Create an offer
createOfferButton.addEventListener("click", async () => {
  const offer = await peerConnection.createOffer();
  await peerConnection.setLocalDescription(offer);

  offerTextarea.value = JSON.stringify(offer);
});

// Set the answer
setAnswerButton.addEventListener("click", async () => {
  const answer = JSON.parse(answerTextarea.value);
  await peerConnection.setRemoteDescription(answer);

  // Add any buffered ICE candidates
  iceCandidates.forEach(candidate => peerConnection.addIceCandidate(candidate));
});

// Handle incoming offer
offerTextarea.addEventListener("input", async () => {
  if (!peerConnection.currentRemoteDescription && offerTextarea.value.trim()) {
    const offer = JSON.parse(offerTextarea.value);
    await peerConnection.setRemoteDescription(offer);

    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    answerTextarea.value = JSON.stringify(answer);
  }
});

// Add ICE candidates manually (for demo purposes)
answerTextarea.addEventListener("input", async () => {
  if (peerConnection.currentRemoteDescription && answerTextarea.value.trim()) {
    const candidate = JSON.parse(answerTextarea.value);
    if (candidate.candidate) {
      await peerConnection.addIceCandidate(candidate);
    }
  }
});
