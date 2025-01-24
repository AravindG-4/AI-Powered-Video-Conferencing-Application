'use client';
import React, { useEffect, useRef } from 'react';
import SimplePeer from 'simple-peer';

const MeetingRoom = () => {
	const videoRef = useRef<HTMLVideoElement>(null);

	useEffect(() => {
		navigator.mediaDevices.getUserMedia({ video: true, audio: true })
			.then(stream => {
				if (videoRef.current) {
					videoRef.current.srcObject = stream;
				}
				const peer = new SimplePeer({
					initiator: true,
					trickle: false,
					stream: stream
				});

				peer.on('signal', data => {
					// Send signal data to server or other peers
				});

				peer.on('stream', remoteStream => {
					// Handle remote stream
				});
			})
			.catch(err => console.error('Error accessing media devices.', err));
	}, []);

	return <video ref={videoRef} autoPlay />;
};

export default MeetingRoom;
