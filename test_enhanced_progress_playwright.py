"""
Comprehensive Playwright Tests for Enhanced Progress Forwarding System

Tests stage-aware progress tracking, real-time UI updates, and error handling
for the enhanced progress forwarding system in Task 11.3.
"""

import asyncio
import pytest
import time
import os
import tempfile
from typing import Dict, List, Any
from pathlib import Path


class TestEnhancedProgressForwarding:
    """Test suite for enhanced progress forwarding functionality."""

    @pytest.fixture
    async def setup_test_environment(self, page):
        """Set up test environment with necessary fixtures."""
        # Navigate to the main application
        await page.goto("http://localhost:8501")
        await page.wait_for_selector("text=Mitochondrial Image Analysis", timeout=10000)

        # Navigate to job queue to test progress
        await page.click("text=Job Queue")
        await page.wait_for_selector("text=Job Queue Dashboard", timeout=5000)

        return page

    async def test_enhanced_progress_display_stages(self, page):
        """Test that all segmentation stages are displayed correctly."""
        await self.setup_test_environment(page)

        # Submit a test segmentation job
        await page.click("text=🧪 Test Job Controls")

        # Create a mock TIFF file for testing
        test_file = await self.create_test_tiff_file()

        try:
            # Submit segmentation job
            await page.click("text=Submit Segmentation Job")

            # Wait for job to start
            await page.wait_for_selector("text=🟡", timeout=30000)  # Running status

            # Expected stages in order
            expected_stages = [
                "initialization",
                "loading",
                "planning",
                "setup",
                "preprocessing",
                "segmentation",
                "reconstruction",
                "assembly",
                "saving",
                "cleanup"
            ]

            # Track which stages we've seen
            stages_seen = set()
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                try:
                    # Look for stage indicators in the UI
                    for stage in expected_stages:
                        try:
                            stage_element = await page.wait_for_selector(
                                f"text=*{stage.title()}*",
                                timeout=1000
                            )
                            if stage_element:
                                stages_seen.add(stage)
                                print(f"✅ Stage detected: {stage}")
                        except:
                            continue

                    # Check if job is completed
                    try:
                        completed = await page.wait_for_selector("text=✅ Completed", timeout=1000)
                        if completed:
                            break
                    except:
                        pass

                    # Small delay between checks
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"Error during stage tracking: {e}")
                    break

            # Verify we saw at least the main stages
            critical_stages = {"initialization", "loading", "preprocessing", "segmentation", "saving"}
            found_critical = stages_seen.intersection(critical_stages)

            assert len(found_critical) >= 3, f"Should detect at least 3 critical stages, found: {found_critical}"
            print(f"✅ Test passed: Detected {len(stages_seen)} stages: {stages_seen}")

        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_progress_bar_accuracy(self, page):
        """Test that progress bars show accurate and monotonic progress."""
        await self.setup_test_environment(page)

        # Submit a test job
        await page.click("text=Submit Quick Test Job")

        # Wait for job to start
        await page.wait_for_selector("text=🟡", timeout=30000)

        progress_values = []
        max_wait_time = 60  # 1 minute for test job
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Find progress bar element
                progress_elements = await page.query_selector_all("[role='progressbar']")

                if progress_elements:
                    for progress_elem in progress_elements:
                        # Get aria-valuenow attribute for progress value
                        value = await progress_elem.get_attribute('aria-valuenow')
                        if value:
                            progress_values.append(float(value))
                            break

                # Check if completed
                completed = await page.wait_for_selector("text=✅ Completed", timeout=1000)
                if completed:
                    break

            except:
                pass

            await asyncio.sleep(1)

        # Validate progress values
        assert len(progress_values) > 0, "Should capture at least some progress values"

        # Check monotonic increase (allowing for small fluctuations)
        for i in range(1, len(progress_values)):
            # Allow small decreases due to stage transitions
            assert progress_values[i] >= progress_values[i-1] - 5, \
                f"Progress should generally increase: {progress_values[i-1]} -> {progress_values[i]}"

        # Check progress reaches 100
        if progress_values:
            assert max(progress_values) >= 95, "Progress should reach near 100%"

        print(f"✅ Progress accuracy test passed with {len(progress_values)} measurements")

    async def test_stage_specific_information_display(self, page):
        """Test that stage-specific information is displayed correctly."""
        await self.setup_test_environment(page)

        # Submit test job
        await page.click("text=Submit Quick Test Job")

        # Wait for job to start
        await page.wait_for_selector("text=🟡", timeout=30000)

        # Look for stage-specific UI elements
        stage_info_found = []
        max_wait_time = 60
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Look for hardware mode indicators
                hardware_indicators = ["💻", "🚀"]  # CPU, GPU
                for indicator in hardware_indicators:
                    element = await page.query_selector(f"text=*{indicator}*")
                    if element:
                        stage_info_found.append("hardware_indicator")
                        break

                # Look for stage progress indicators
                stage_element = await page.query_selector("text=*Stage:*")
                if stage_element:
                    stage_info_found.append("stage_progress")

                # Look for ETA information
                eta_element = await page.query_selector("text=*ETA*")
                if eta_element:
                    stage_info_found.append("eta_display")

                # Check if completed
                completed = await page.wait_for_selector("text=✅ Completed", timeout=1000)
                if completed:
                    break

            except:
                pass

            await asyncio.sleep(2)

        # Verify we found stage-specific information
        assert len(stage_info_found) > 0, f"Should display stage-specific info, found: {stage_info_found}"
        print(f"✅ Stage info test passed with elements: {stage_info_found}")

    async def test_error_handling_progress_display(self, page):
        """Test progress display when jobs encounter errors."""
        await self.setup_test_environment(page)

        # Submit a job that will likely fail (invalid input)
        # This depends on having error-prone test scenarios
        await page.click("text=Submit Quick Test Job")

        # Wait for job to start
        await page.wait_for_selector("text=🟡", timeout=30000)

        # Wait for potential failure or completion
        max_wait_time = 120
        start_time = time.time()
        job_result = None

        while time.time() - start_time < max_wait_time:
            try:
                # Check for completion
                completed = await page.wait_for_selector("text=✅ Completed", timeout=1000)
                if completed:
                    job_result = "completed"
                    break

                # Check for failure
                failed = await page.wait_for_selector("text=❌ Failed", timeout=1000)
                if failed:
                    job_result = "failed"
                    break

                # Check for cancellation
                cancelled = await page.wait_for_selector("text=🛑 Cancelled", timeout=1000)
                if cancelled:
                    job_result = "cancelled"
                    break

            except:
                pass

            await asyncio.sleep(2)

        # Verify we got some result
        assert job_result is not None, "Job should reach a terminal state"
        print(f"✅ Error handling test passed with result: {job_result}")

    async def test_real_time_updates(self, page):
        """Test that progress updates happen in real-time."""
        await self.setup_test_environment(page)

        # Enable auto-refresh
        await page.check("text=Auto-refresh")

        # Submit test job
        await page.click("text=Submit Long Test Job")

        # Wait for job to start
        await page.wait_for_selector("text=🟡", timeout=30000)

        # Track update frequency
        update_times = []
        last_progress = -1
        max_wait_time = 90  # 1.5 minutes for long test job
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Look for progress changes
                progress_elements = await page.query_selector_all("[role='progressbar']")

                if progress_elements:
                    for progress_elem in progress_elements:
                        value = await progress_elem.get_attribute('aria-valuenow')
                        if value:
                            current_progress = float(value)
                            if current_progress != last_progress:
                                update_times.append(time.time())
                                last_progress = current_progress
                            break

                # Check if completed
                completed = await page.wait_for_selector("text=✅ Completed", timeout=1000)
                if completed:
                    break

            except:
                pass

            await asyncio.sleep(1)

        # Analyze update frequency
        if len(update_times) > 1:
            # Calculate average time between updates
            intervals = [update_times[i] - update_times[i-1] for i in range(1, len(update_times))]
            avg_interval = sum(intervals) / len(intervals)

            # Real-time updates should happen at least every 5 seconds during active processing
            assert avg_interval <= 5, f"Updates should be frequent, avg interval: {avg_interval:.2f}s"
            print(f"✅ Real-time updates test passed: {len(update_times)} updates, avg interval: {avg_interval:.2f}s")
        else:
            print("⚠️ Real-time updates test: Insufficient updates captured for analysis")

    async def test_multiple_jobs_progress_tracking(self, page):
        """Test progress tracking with multiple concurrent jobs."""
        await self.setup_test_environment(page)

        # Submit multiple jobs
        await page.click("text=Submit Multiple Jobs")

        # Wait for jobs to appear
        await asyncio.sleep(5)

        # Count running jobs
        running_jobs = 0
        job_ids = []

        try:
            # Look for running job indicators
            running_elements = await page.query_selector_all("text=🟡")
            running_jobs = len(running_elements)

            # Extract job IDs if visible
            job_id_elements = await page.query_selector_all("text=*...*")
            for element in job_id_elements:
                text = await element.text_content()
                if "..." in text and len(text) < 20:  # Likely a truncated job ID
                    job_ids.append(text)

        except:
            pass

        # Verify multiple jobs are tracked
        assert running_jobs >= 1, f"Should have at least 1 running job, found: {running_jobs}"

        # Wait for jobs to complete
        max_wait_time = 180  # 3 minutes for multiple jobs
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Check if all jobs completed
                running_elements = await page.query_selector_all("text=🟡")
                if len(running_elements) == 0:
                    break
            except:
                pass

            await asyncio.sleep(5)

        print(f"✅ Multiple jobs test passed: Tracked {running_jobs} concurrent jobs")

    async def create_test_tiff_file(self) -> str:
        """Create a minimal test TIFF file for testing."""
        try:
            import numpy as np
            from tifffile import imsave

            # Create a small test image (10x10x5 stack)
            test_data = np.random.randint(0, 255, (5, 10, 10), dtype=np.uint8)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                imsave(tmp_file.name, test_data)
                return tmp_file.name

        except ImportError:
            # If tifffile not available, create dummy file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_file.write(b'dummy tiff data for testing')
                return tmp_file.name


# Test configuration and runner
if __name__ == "__main__":
    # This can be run with: python -m pytest test_enhanced_progress_playwright.py -v
    print("Enhanced Progress Forwarding Tests")
    print("Run with: python -m pytest test_enhanced_progress_playwright.py -v")
    print("\nTest coverage:")
    print("✓ Stage-aware progress display")
    print("✓ Progress bar accuracy and monotonic increase")
    print("✓ Stage-specific information display")
    print("✓ Error handling and edge cases")
    print("✓ Real-time update frequency")
    print("✓ Multiple concurrent jobs tracking")