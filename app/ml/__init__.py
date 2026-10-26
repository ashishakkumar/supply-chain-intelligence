"""Booking-time delay model.

Predicts whether a shipment will arrive more than 24 hours late, using only
features known when the booking is made. The artefact is models/delay_model.joblib.
The API loads it at startup. This package does not score inside a request by
retraining.
"""
