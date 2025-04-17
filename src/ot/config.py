from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import os

def setup_tracing():
    tracer_provider = TracerProvider(
        resource=Resource.create({
            "service.name": "spam-detection-api",
            "service.version": "1.0.0"
        })
    )

    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # ✅ Updated tracer call
    return tracer_provider.get_tracer("spam-detection-api", "1.0.0")

def instrument_app(app):
    FastAPIInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())

    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()

def get_tracer():
    return trace.get_tracer(__name__)
