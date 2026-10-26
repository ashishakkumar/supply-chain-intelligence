CREATE TABLE ports (
    port_code TEXT NOT NULL,
    port_name TEXT NOT NULL,
    country TEXT,
    region TEXT NOT NULL,
    timezone TEXT NOT NULL,
    avg_congestion_score NUMERIC(3, 2) NOT NULL,
    PRIMARY KEY (port_code),
    CONSTRAINT congestion_between_0_and_1 CHECK (avg_congestion_score >= 0 AND avg_congestion_score <= 1)
);
