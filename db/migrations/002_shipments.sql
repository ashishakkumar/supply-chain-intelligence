CREATE TABLE shipments (
    shipment_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    origin_port TEXT NOT NULL,
    destination_port TEXT NOT NULL,
    vessel_id TEXT NOT NULL,
    booking_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    planned_departure TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    actual_departure TIMESTAMP WITHOUT TIME ZONE,
    planned_arrival TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    actual_arrival TIMESTAMP WITHOUT TIME ZONE,
    container_count INTEGER NOT NULL,
    cargo_type TEXT,
    weight_tons NUMERIC(10, 2),
    status TEXT NOT NULL,
    actual_delay_hours FLOAT,
    on_time_flag BOOLEAN,
    route_key TEXT NOT NULL,
    transit_days_planned FLOAT,
    transit_days_actual FLOAT,
    PRIMARY KEY (shipment_id),
    CONSTRAINT weight_non_negative CHECK (weight_tons >= 0),
    CONSTRAINT container_non_negative CHECK (container_count >= 0),
    CONSTRAINT status_canonical CHECK (status IN ('DELIVERED', 'DELAYED', 'CANCELLED', 'COMPLETED', 'UNKNOWN')),
    FOREIGN KEY(origin_port) REFERENCES ports (port_code),
    FOREIGN KEY(destination_port) REFERENCES ports (port_code)
);

CREATE INDEX ix_shipments_route ON shipments (origin_port, destination_port);

CREATE INDEX ix_shipments_booking ON shipments (booking_date);
