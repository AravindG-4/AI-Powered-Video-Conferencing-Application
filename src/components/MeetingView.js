// src/components/MeetingView.js

import React from 'react';
import ParticipantView from './ParticipantView';
import Controls from './Controls';

const MeetingView = () => {
    return (
        <div>
            <h1>Video Conference</h1>
            <ParticipantView />
            <Controls />
        </div>
    );
};

export default MeetingView;
