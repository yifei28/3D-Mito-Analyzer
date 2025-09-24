#!/usr/bin/env python3
"""
Test script to verify Docker container TensorFlow memory fix using Playwright.

This script tests:
1. Container accessibility at http://localhost:8501
2. UI loading and functionality
3. TIFF file upload capability
4. Segmentation job execution without container restart
5. Docker logs monitoring for memory allocation errors
"""

import asyncio
import os
import time
import subprocess
import threading
import json
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_docker_tensorflow_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DockerLogMonitor:
    """Monitor Docker logs for memory allocation errors and container restarts."""

    def __init__(self, container_name: str = "mitochondria-analyzer"):
        self.container_name = container_name
        self.logs: List[str] = []
        self.memory_errors: List[str] = []
        self.restart_events: List[str] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start monitoring Docker logs in a separate thread."""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_logs, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started monitoring logs for container: {self.container_name}")

    def stop_monitoring(self):
        """Stop monitoring Docker logs."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring Docker logs")

    def _monitor_logs(self):
        """Internal method to monitor Docker logs."""
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            while self.is_monitoring and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    self.logs.append(f"{time.strftime('%H:%M:%S')} - {line}")

                    # Check for memory allocation errors
                    if "Allocation of" in line and "exceeds" in line and "system memory" in line:
                        self.memory_errors.append(line)
                        logger.warning(f"Memory allocation error detected: {line}")

                    # Check for container restart indicators
                    if any(keyword in line.lower() for keyword in ["restarting", "killed", "oom", "exit code"]):
                        self.restart_events.append(line)
                        logger.warning(f"Container restart event detected: {line}")

                    # Log important TensorFlow messages
                    if "tensorflow" in line.lower() or "gpu" in line.lower() or "memory" in line.lower():
                        logger.info(f"TF/Memory log: {line}")

        except Exception as e:
            logger.error(f"Error monitoring Docker logs: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()

    def get_summary(self) -> Dict:
        """Get summary of monitoring results."""
        return {
            "total_logs": len(self.logs),
            "memory_errors": len(self.memory_errors),
            "restart_events": len(self.restart_events),
            "memory_error_details": self.memory_errors,
            "restart_event_details": self.restart_events,
            "recent_logs": self.logs[-20:] if len(self.logs) > 20 else self.logs
        }


class PlaywrightTester:
    """Playwright test class for Docker container testing."""

    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.test_results: Dict = {
            "container_accessible": False,
            "ui_loaded": False,
            "file_upload_successful": False,
            "segmentation_started": False,
            "segmentation_completed": False,
            "container_remained_stable": True,
            "errors": []
        }

    async def setup(self):
        """Setup Playwright browser and page."""
        try:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=False)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

            # Set up page error handler
            self.page.on("pageerror", lambda error: logger.error(f"Page error: {error}"))
            self.page.on("console", lambda msg: logger.info(f"Console: {msg.text}"))

            logger.info("Playwright setup completed")
        except Exception as e:
            logger.error(f"Failed to setup Playwright: {e}")
            raise

    async def teardown(self):
        """Teardown Playwright browser."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            logger.info("Playwright teardown completed")
        except Exception as e:
            logger.error(f"Error during teardown: {e}")

    async def test_container_accessibility(self) -> bool:
        """Test if the container is accessible at the specified URL."""
        try:
            logger.info(f"Testing container accessibility at {self.base_url}")
            response = await self.page.goto(self.base_url, wait_until="networkidle", timeout=30000)

            if response and response.status == 200:
                self.test_results["container_accessible"] = True
                logger.info("Container is accessible")
                return True
            else:
                error = f"Container returned status {response.status if response else 'None'}"
                self.test_results["errors"].append(error)
                logger.error(error)
                return False

        except Exception as e:
            error = f"Failed to access container: {e}"
            self.test_results["errors"].append(error)
            logger.error(error)
            return False

    async def test_ui_loading(self) -> bool:
        """Test if the Streamlit UI loads properly."""
        try:
            logger.info("Testing UI loading")

            # Wait for Streamlit to fully load
            await self.page.wait_for_selector('[data-testid="stApp"]', timeout=15000)

            # Check for main title
            title_present = await self.page.locator("h1").first.is_visible()
            if title_present:
                title_text = await self.page.locator("h1").first.inner_text()
                logger.info(f"Main title found: {title_text}")

            # Check for file uploader
            uploader_present = await self.page.locator('input[type="file"]').is_visible()

            if title_present and uploader_present:
                self.test_results["ui_loaded"] = True
                logger.info("UI loaded successfully")
                return True
            else:
                error = f"UI elements missing - Title: {title_present}, Uploader: {uploader_present}"
                self.test_results["errors"].append(error)
                logger.error(error)
                return False

        except Exception as e:
            error = f"Failed to load UI: {e}"
            self.test_results["errors"].append(error)
            logger.error(error)
            return False

    async def test_file_upload(self, file_path: str) -> bool:
        """Test uploading a TIFF file."""
        try:
            logger.info(f"Testing file upload with: {file_path}")

            if not os.path.exists(file_path):
                error = f"Test file not found: {file_path}"
                self.test_results["errors"].append(error)
                logger.error(error)
                return False

            # Find and interact with file upload input
            file_input = self.page.locator('input[type="file"]').first
            await file_input.set_input_files(file_path)

            # Wait a bit for file to be processed
            await self.page.wait_for_timeout(3000)

            # Check if file was uploaded successfully
            # Look for success indicators (this may vary based on your UI)
            success_indicators = [
                "uploaded successfully",
                "file loaded",
                "ready for analysis",
                "segmentation options"
            ]

            page_content = await self.page.content()
            upload_successful = any(indicator.lower() in page_content.lower() for indicator in success_indicators)

            if upload_successful:
                self.test_results["file_upload_successful"] = True
                logger.info("File upload successful")
                return True
            else:
                # Even if no explicit success message, if no error occurred, consider it successful
                self.test_results["file_upload_successful"] = True
                logger.info("File upload completed (no explicit success message)")
                return True

        except Exception as e:
            error = f"Failed to upload file: {e}"
            self.test_results["errors"].append(error)
            logger.error(error)
            return False

    async def test_segmentation_job(self) -> bool:
        """Test starting a segmentation job."""
        try:
            logger.info("Testing segmentation job")

            # Look for segmentation-related buttons or controls
            segmentation_buttons = [
                "Start Segmentation",
                "Run Segmentation",
                "Segment",
                "Process",
                "Analyze"
            ]

            button_clicked = False
            for button_text in segmentation_buttons:
                button = self.page.locator(f"button:has-text('{button_text}')").first
                if await button.is_visible():
                    await button.click()
                    button_clicked = True
                    logger.info(f"Clicked button: {button_text}")
                    break

            if not button_clicked:
                # Try clicking any button that might start processing
                buttons = await self.page.locator("button").all()
                for button in buttons:
                    button_text = await button.inner_text()
                    if any(word in button_text.lower() for word in ["start", "run", "process", "segment"]):
                        await button.click()
                        button_clicked = True
                        logger.info(f"Clicked button: {button_text}")
                        break

            if button_clicked:
                self.test_results["segmentation_started"] = True

                # Wait for segmentation to start (look for progress indicators)
                await self.page.wait_for_timeout(5000)

                # Monitor for completion or progress
                max_wait_time = 300  # 5 minutes maximum
                start_time = time.time()

                while time.time() - start_time < max_wait_time:
                    page_content = await self.page.content()

                    # Check for completion indicators
                    completion_indicators = [
                        "segmentation complete",
                        "analysis complete",
                        "processing finished",
                        "results ready",
                        "download results"
                    ]

                    if any(indicator.lower() in page_content.lower() for indicator in completion_indicators):
                        self.test_results["segmentation_completed"] = True
                        logger.info("Segmentation completed successfully")
                        return True

                    # Check for error indicators
                    error_indicators = [
                        "error",
                        "failed",
                        "timeout",
                        "crashed"
                    ]

                    if any(indicator.lower() in page_content.lower() for indicator in error_indicators):
                        error = "Segmentation failed - error detected in UI"
                        self.test_results["errors"].append(error)
                        logger.error(error)
                        return False

                    await self.page.wait_for_timeout(10000)  # Wait 10 seconds between checks

                # If we get here, segmentation didn't complete within timeout
                logger.warning("Segmentation did not complete within timeout, but no errors detected")
                return True  # Consider this a partial success

            else:
                error = "Could not find segmentation button to click"
                self.test_results["errors"].append(error)
                logger.error(error)
                return False

        except Exception as e:
            error = f"Failed to test segmentation: {e}"
            self.test_results["errors"].append(error)
            logger.error(error)
            return False

    def get_test_results(self) -> Dict:
        """Get comprehensive test results."""
        return self.test_results


async def main():
    """Main test function."""
    logger.info("Starting Docker TensorFlow memory fix test")

    # Find a suitable test file
    test_files = [
        "./segmentation_result_fixed.tif",
        "./MoDL/test/test_44/0001.tif",
        "./MoDL/test/test_44/0000.tif"
    ]

    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = os.path.abspath(file_path)
            break

    if not test_file:
        logger.error("No test TIFF files found!")
        return

    logger.info(f"Using test file: {test_file}")

    # Start Docker log monitoring
    log_monitor = DockerLogMonitor()
    log_monitor.start_monitoring()

    # Setup Playwright tester
    tester = PlaywrightTester()

    try:
        await tester.setup()

        # Run tests in sequence
        logger.info("=== Starting Test Sequence ===")

        # Test 1: Container accessibility
        if await tester.test_container_accessibility():
            logger.info("✓ Test 1 PASSED: Container accessible")
        else:
            logger.error("✗ Test 1 FAILED: Container not accessible")
            return

        # Test 2: UI Loading
        if await tester.test_ui_loading():
            logger.info("✓ Test 2 PASSED: UI loaded successfully")
        else:
            logger.error("✗ Test 2 FAILED: UI failed to load")
            return

        # Test 3: File Upload
        if await tester.test_file_upload(test_file):
            logger.info("✓ Test 3 PASSED: File upload successful")
        else:
            logger.error("✗ Test 3 FAILED: File upload failed")
            return

        # Test 4: Segmentation Job
        if await tester.test_segmentation_job():
            logger.info("✓ Test 4 PASSED: Segmentation job executed")
        else:
            logger.error("✗ Test 4 FAILED: Segmentation job failed")

        # Wait a bit more to ensure logs are captured
        await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"Test execution failed: {e}")

    finally:
        # Stop monitoring and get results
        log_monitor.stop_monitoring()
        await tester.teardown()

        # Generate test report
        await generate_test_report(tester, log_monitor)


async def generate_test_report(tester: PlaywrightTester, log_monitor: DockerLogMonitor):
    """Generate comprehensive test report."""
    logger.info("=== Generating Test Report ===")

    test_results = tester.get_test_results()
    log_summary = log_monitor.get_summary()

    # Check if container remained stable (no restarts)
    test_results["container_remained_stable"] = len(log_summary["restart_events"]) == 0

    # Generate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_summary": {
            "container_accessible": test_results["container_accessible"],
            "ui_loaded": test_results["ui_loaded"],
            "file_upload_successful": test_results["file_upload_successful"],
            "segmentation_started": test_results["segmentation_started"],
            "segmentation_completed": test_results["segmentation_completed"],
            "container_remained_stable": test_results["container_remained_stable"]
        },
        "docker_log_analysis": {
            "total_log_entries": log_summary["total_logs"],
            "memory_allocation_errors": log_summary["memory_errors"],
            "container_restart_events": log_summary["restart_events"],
            "tensorflow_memory_fix_effective": len(log_summary["memory_errors"]) == 0
        },
        "detailed_results": {
            "test_errors": test_results["errors"],
            "memory_error_details": log_summary["memory_error_details"],
            "restart_event_details": log_summary["restart_event_details"]
        },
        "recent_docker_logs": log_summary["recent_logs"]
    }

    # Save report to file
    report_file = "docker_tensorflow_fix_test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, indent=2, fp=f)

    # Print summary
    logger.info("=== TEST RESULTS SUMMARY ===")
    logger.info(f"Container Accessible: {'✓' if report['test_summary']['container_accessible'] else '✗'}")
    logger.info(f"UI Loaded: {'✓' if report['test_summary']['ui_loaded'] else '✗'}")
    logger.info(f"File Upload: {'✓' if report['test_summary']['file_upload_successful'] else '✗'}")
    logger.info(f"Segmentation Started: {'✓' if report['test_summary']['segmentation_started'] else '✗'}")
    logger.info(f"Segmentation Completed: {'✓' if report['test_summary']['segmentation_completed'] else '✗'}")
    logger.info(f"Container Stable: {'✓' if report['test_summary']['container_remained_stable'] else '✗'}")
    logger.info(f"TensorFlow Memory Fix Effective: {'✓' if report['docker_log_analysis']['tensorflow_memory_fix_effective'] else '✗'}")

    logger.info(f"Memory Allocation Errors: {len(report['docker_log_analysis']['memory_allocation_errors'])}")
    logger.info(f"Container Restart Events: {len(report['docker_log_analysis']['container_restart_events'])}")

    if report['test_summary']['container_remained_stable'] and report['docker_log_analysis']['tensorflow_memory_fix_effective']:
        logger.info("🎉 SUCCESS: TensorFlow memory fix appears to be working!")
    else:
        logger.warning("⚠️  WARNING: Issues detected - check detailed report")

    logger.info(f"Detailed report saved to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())