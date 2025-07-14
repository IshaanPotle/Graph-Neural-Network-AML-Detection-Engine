-- AML Engine Database Schema
-- PostgreSQL initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_hash VARCHAR(255) UNIQUE NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    amount DECIMAL(20, 8) NOT NULL,
    currency VARCHAR(10) DEFAULT 'BTC',
    timestamp TIMESTAMP NOT NULL,
    block_height INTEGER,
    fee DECIMAL(20, 8),
    risk_score DECIMAL(5, 4),
    is_suspicious BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Wallets table
CREATE TABLE IF NOT EXISTS wallets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(255) UNIQUE NOT NULL,
    wallet_type VARCHAR(50),
    risk_score DECIMAL(5, 4) DEFAULT 0.0,
    total_volume DECIMAL(20, 8) DEFAULT 0.0,
    transaction_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    is_suspicious BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    entity_id UUID,
    entity_type VARCHAR(50),
    risk_score DECIMAL(5, 4),
    status VARCHAR(20) DEFAULT 'open',
    assigned_to UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    metadata JSONB
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    prediction DECIMAL(5, 4) NOT NULL,
    confidence DECIMAL(5, 4),
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_hash ON transactions(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_transactions_from ON transactions(from_address);
CREATE INDEX IF NOT EXISTS idx_transactions_to ON transactions(to_address);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_risk ON transactions(risk_score);

CREATE INDEX IF NOT EXISTS idx_wallets_address ON wallets(address);
CREATE INDEX IF NOT EXISTS idx_wallets_risk ON wallets(risk_score);
CREATE INDEX IF NOT EXISTS idx_wallets_type ON wallets(wallet_type);

CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_entity ON predictions(entity_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(created_at);

CREATE INDEX IF NOT EXISTS idx_performance_service ON performance_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_transactions_search ON transactions USING gin(to_tsvector('english', transaction_hash || ' ' || from_address || ' ' || to_address));
CREATE INDEX IF NOT EXISTS idx_alerts_search ON alerts USING gin(to_tsvector('english', title || ' ' || description));

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@aml-engine.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8i', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW high_risk_transactions AS
SELECT * FROM transactions 
WHERE risk_score > 0.7 
ORDER BY risk_score DESC, timestamp DESC;

CREATE OR REPLACE VIEW suspicious_wallets AS
SELECT * FROM wallets 
WHERE is_suspicious = true OR risk_score > 0.8
ORDER BY risk_score DESC;

CREATE OR REPLACE VIEW recent_alerts AS
SELECT a.*, u.username as assigned_username
FROM alerts a
LEFT JOIN users u ON a.assigned_to = u.id
WHERE a.status = 'open'
ORDER BY a.created_at DESC;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_wallet_risk_score(wallet_address VARCHAR, new_risk_score DECIMAL)
RETURNS VOID AS $$
BEGIN
    UPDATE wallets 
    SET risk_score = new_risk_score, 
        updated_at = CURRENT_TIMESTAMP,
        is_suspicious = (new_risk_score > 0.7)
    WHERE address = wallet_address;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_alert(
    p_alert_type VARCHAR,
    p_severity VARCHAR,
    p_title VARCHAR,
    p_description TEXT,
    p_entity_id VARCHAR,
    p_entity_type VARCHAR,
    p_risk_score DECIMAL
) RETURNS UUID AS $$
DECLARE
    alert_id UUID;
BEGIN
    INSERT INTO alerts (alert_type, severity, title, description, entity_id, entity_type, risk_score)
    VALUES (p_alert_type, p_severity, p_title, p_description, p_entity_id, p_entity_type, p_risk_score)
    RETURNING id INTO alert_id;
    
    RETURN alert_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your security requirements)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aml_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO aml_user; 