"""Tool-calling assistant over the curated shipment data.

CLI only. The model must call tools for facts. query_shipments and
get_route_stats read Postgres. predict_delay is the only tool that calls
the HTTP API. Off-domain questions are refused, tool rounds are capped at 5,
and a factual sentence with no tool result is replaced.
"""
