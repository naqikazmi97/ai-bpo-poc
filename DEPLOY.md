# EC2 Deployment Setup

## IAM Role for EC2
Attach a role with these policies:
- AmazonTranscribeFullAccess (or scoped: transcribe:StartStreamTranscription)
- AmazonPollyFullAccess (or scoped: polly:SynthesizeSpeech)
- AmazonBedrockFullAccess (or scoped: bedrock:InvokeModelWithResponseStream, bedrock:InvokeModel)
- AmazonDynamoDBFullAccess (or scoped to your table)

## DynamoDB Table
aws dynamodb create-table \
  --table-name voice-bot-sessions \
  --attribute-definitions AttributeName=session_id,AttributeType=S \
  --key-schema AttributeName=session_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

Enable TTL on the table (auto-deletes sessions after 24h):
aws dynamodb update-time-to-live \
  --table-name voice-bot-sessions \
  --time-to-live-specification "Enabled=true, AttributeName=ttl" \
  --region us-east-1

## EC2 Bootstrap (Amazon Linux 2023)
sudo dnf update -y
sudo dnf install python3.11 python3.11-pip git -y
pip3.11 install -r requirements.txt

## Systemd service
cat > /etc/systemd/system/voicebot.service << 'EOF'
[Unit]
Description=Voice Bot FastAPI Server
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/voicebot
ExecStart=/usr/local/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment=AWS_DEFAULT_REGION=us-east-1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable voicebot
sudo systemctl start voicebot

## ALB Setup
1. Create Target Group:
   - Type: Instance
   - Protocol: HTTP / Port: 8000
   - Health check: GET /health

2. Create ALB:
   - Listener: HTTPS 443
   - WebSocket supported natively — no extra config needed
   - Forward to Target Group

3. Security Group for EC2:
   - Inbound: TCP 8000 from ALB security group only
   - Outbound: all (for AWS API calls to Transcribe, Bedrock, Polly, DynamoDB)

4. Security Group for ALB:
   - Inbound: TCP 443 from 0.0.0.0/0
   - Outbound: TCP 8000 to EC2 security group

## Auto Scaling Group
- Min: 2  (always-on, no cold start gap)
- Max: 10
- Scale-out trigger: CPU > 70% for 2 minutes
- Scale-in  trigger: CPU < 30% for 10 minutes

## Frontend
- Update WS_URL in voicebot-frontend.html to your ALB DNS or custom domain
- Host on S3 + CloudFront, or serve from FastAPI directly:
  app.mount("/", StaticFiles(directory="static", html=True), name="static")

## HTTPS / WSS
ALB terminates TLS. Frontend connects via wss://.
Backend handles plain ws:// only.

## Session data in DynamoDB
Each session record contains:
- session_id      (partition key)
- history         (list of {role, content} messages)
- slots           (extracted entities — written at session end)
- updated_at      (ISO timestamp)
- ttl             (Unix timestamp — auto-deleted after 24h)
