# 🐳 Docker Deployment Guide for Windows PC

This guide will help you deploy the Mitochondria Analyzer on your Windows PC using Docker.

## 📋 Prerequisites

### 1. Install Docker Desktop for Windows
- Download from: https://docs.docker.com/desktop/windows/install/
- System Requirements:
  - Windows 10 64-bit: Pro, Enterprise, or Education (Build 19041 or higher)
  - WSL 2 backend enabled
  - At least 4GB RAM (8GB recommended)
  - Virtualization enabled in BIOS

### 2. Enable WSL 2 (if not already enabled)
```powershell
# Run in PowerShell as Administrator
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart Windows, then set WSL 2 as default
wsl --set-default-version 2
```

### 3. Install Git for Windows
- Download from: https://git-scm.com/download/win

## 🚀 Deployment Steps

### Step 1: Clone the Repository
```bash
# Open Command Prompt or PowerShell
git clone https://github.com/your-username/mito-analyzer.git
cd mito-analyzer
```

### Step 2: Prepare Data Directories
```bash
# Create necessary directories for data volumes
mkdir data\raw
mkdir data\segmented
mkdir data\analyzed
mkdir logs

# Copy your TIFF images to data\raw\
# Example:
# copy "C:\path\to\your\images\*.tif" data\raw\
```

### Step 3: Build and Run with Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up --build

# To run in background (detached mode)
docker-compose up -d --build
```

### Step 4: Alternative - Direct Docker Commands
```bash
# Build the image
docker build -t mito-analyzer .

# Run the container
docker run -d \
  --name mitochondria-analyzer \
  -p 8501:8501 \
  -v ${PWD}/data/raw:/home/app/data/raw \
  -v ${PWD}/data/segmented:/home/app/data/segmented \
  -v ${PWD}/data/analyzed:/home/app/data/analyzed \
  -v ${PWD}/logs:/home/app/logs \
  mito-analyzer
```

### Step 5: Access the Application
- Open your web browser
- Navigate to: http://localhost:8501
- The Mitochondria Analyzer interface should load

## 📁 Volume Mounts Explained

| Host Directory | Container Directory | Purpose |
|----------------|-------------------|---------|
| `./data/raw` | `/app/data/raw` | Input TIFF images |
| `./data/segmented` | `/app/data/segmented` | Segmentation results |
| `./data/analyzed` | `/app/data/analyzed` | Analysis results |
| `./logs` | `/app/logs` | Application logs |
| `./MoDL/model` | `/app/MoDL/model` | **Required**: MoDL model weights (~456MB) |

## 🔧 Configuration

### Resource Limits
The `docker-compose.yml` file includes resource limits:
- CPU: 4 cores max, 2 cores reserved
- Memory: 8GB max, 4GB reserved

Adjust these based on your Windows PC specifications by editing `docker-compose.yml`.

### Port Configuration
- Default port: 8501
- To change port: Edit the `ports` section in `docker-compose.yml`
- Example for port 8080: `"8080:8501"`

## 📊 Performance Optimization for Windows

### 1. WSL 2 Memory Allocation
Create/edit `C:\Users\<username>\.wslconfig`:
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

### 2. Docker Desktop Settings
- Go to Docker Desktop → Settings → Resources
- Adjust Memory and CPU allocation
- **CRITICAL**: Set Memory to at least 10-12GB RAM, 4+ CPU cores
- This is required for TensorFlow inference during segmentation

### 3. File System Performance
- Store project files on your main Windows drive (C:)
- Avoid network drives for better I/O performance

## 🛠️ Management Commands

### View Running Containers
```bash
docker ps
```

### View Logs
```bash
# Using Docker Compose
docker-compose logs -f

# Using Docker directly
docker logs mitochondria-analyzer -f
```

### Stop the Application
```bash
# Using Docker Compose
docker-compose down

# Using Docker directly
docker stop mitochondria-analyzer
docker rm mitochondria-analyzer
```

### Update the Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
```bash
# Check what's using port 8501
netstat -ano | findstr :8501

# Kill the process or use a different port
```

#### 2. Out of Memory Errors
- Increase Docker Desktop memory allocation
- Reduce resource limits in `docker-compose.yml`
- Close other applications to free up RAM

#### 3. Volume Mount Issues
- Ensure directories exist before starting container
- Check file permissions
- Use full Windows paths in volume mounts

#### 4. WSL 2 Issues
```bash
# Restart WSL
wsl --shutdown
wsl

# Update WSL
wsl --update
```

#### 5. Container Won't Start
```bash
# Check container logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker system prune -a
docker-compose up --build
```

## 📈 Health Check

The container includes a health check that monitors the Streamlit application:
- Check interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds
- Retries: 3

View health status:
```bash
docker ps
# Look for "healthy" status
```

## 🔒 Security Notes

- The application runs as non-root user `app`
- Only necessary ports are exposed
- Data directories have restricted permissions
- No sensitive environment variables are included

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify Docker Desktop is running
3. Ensure WSL 2 is properly configured
4. Check Windows Firewall settings
5. Verify hardware requirements are met

## 🎯 Next Steps

After successful deployment:

1. Upload your TIFF images to `data/raw/`
2. Use the web interface to run segmentation
3. Analyze results in the Analysis tab
4. Export results as needed
5. Access logs in the `logs/` directory

Happy analyzing! 🔬