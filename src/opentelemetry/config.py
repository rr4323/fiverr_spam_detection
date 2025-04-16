from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import os

def setup_tracing(service_name: str = "spam-detection-service"):
    """Initialize OpenTelemetry tracing with the given service name."""
    # Create resource with service name
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development")
    })
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    
    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return trace.get_tracer(__name__)

def instrument_fastapi(app):
    """Instrument FastAPI application with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()

def get_tracer():
    """Get the current tracer instance."""
    return trace.get_tracer(__name__) 