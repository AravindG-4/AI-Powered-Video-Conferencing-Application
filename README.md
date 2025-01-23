# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app) and simple-peer webrtc and videosdk.live/react-sdk

# Why use Simple-Peer?
If you’re developing a browser-based peer-to-peer application and looking for a compact, user-friendly solution, you might want to consider using Simple-Peer. It is made explicitly for developing peer-to-peer apps in the browser. If you’re seeking a quick and effective method of transferring data directly between peers, Simple-Peer is another excellent option. The direct peer-to-peer links it creates allow for data’s speedy and effective movement.

Here’s a brief overview of how Simple-Peer works:

One of the peers develops an offer using the RTCPeerConnection API, outlining the different media and data types it can exchange.
The other peer receives the offer using a signaling server, which is used to communicate the metadata required to create the connection. Any approach, including WebSockets or HTTP, can be used to implement the signaling server.
The other peer accepts the offer and uses the RTCPeerConnection API to construct a response. The answer explains the different media and data kinds it can interchange.
The response is relayed to the initial peer using the signaling server.
After exchanging offers and answers, the two peers can connect and exchange media and data directly.
