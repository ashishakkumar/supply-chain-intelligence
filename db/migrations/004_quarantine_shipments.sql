CREATE TABLE quarantine_shipments (
    id SERIAL NOT NULL,
    shipment_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw JSONB NOT NULL,
    PRIMARY KEY (id)
);
