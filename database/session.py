from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import settings

# Database Connection Pool Configuration
# Using create_engine with pool_size and max_overflow for production readiness
engine = create_engine(
    settings.DATABASE_URL, 
    pool_pre_ping=True, # Check connection before using from pool
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency to get a database session per request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
