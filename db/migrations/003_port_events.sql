CREATE TABLE port_events (
    id SERIAL NOT NULL,
    event_id TEXT NOT NULL,
    event_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    port_code TEXT NOT NULL,
    vessel_id TEXT NOT NULL,
    event_type TEXT,
    delay_minutes INTEGER NOT NULL,
    notes TEXT,
    PRIMARY KEY (id),
    CONSTRAINT delay_minutes_non_negative CHECK (delay_minutes >= 0),
    FOREIGN KEY(port_code) REFERENCES ports (port_code)
);
