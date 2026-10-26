CREATE TABLE quarantine_events (
    id SERIAL NOT NULL,
    event_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw JSONB NOT NULL,
    PRIMARY KEY (id)
);
