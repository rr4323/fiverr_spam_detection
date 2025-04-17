from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import os

def setup_frontend_tracing():
    """Setup OpenTelemetry tracing for the frontend"""
    # Get environment variables
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Create resource
    resource = Resource.create({
        "service.name": "spam-detection-frontend",
        "service.version": "1.0.0",
        "deployment.environment": environment
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Create OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otel_endpoint,
        insecure=True
    )
    
    # Create span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    
    # Add span processor to tracer provider
    tracer_provider.add_span_processor(span_processor)
    
    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument requests
    RequestsInstrumentor().instrument()
    
    return trace.get_tracer("spam-detection-frontend") 