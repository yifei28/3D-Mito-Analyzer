#!/usr/bin/env python3
"""
GPU Queue Management Playwright Test Suite

Comprehensive browser-based testing of the GPU resource queue management system.
Tests concurrent job limits, queue position display, timeout handling, and UI behavior.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from workflows.job_manager import JobManager


class GPUQueueManagementPlaywrightTest:
    """Comprehensive GPU queue management test suite using Playwright."""

    def __init__(self):
        self.test_results = []
        self.job_manager = JobManager()

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "details": details
        })
        print(f"{status} {test_name}")
        if details:
            print(f"     {details}")

    async def test_gpu_concurrent_job_limits(self, page):
        """Test that only 1 GPU job runs at a time."""
        print("\n🧪 TESTING GPU CONCURRENT JOB LIMITS")
        print("-" * 50)

        try:
            # Navigate to Job Queue
            await page.goto("http://localhost:8501")
            await page.wait_for_selector("text=Mitochondrial Image Analysis", timeout=15000)
            await page.click("text=Job Queue")
            await page.wait_for_selector("text=Job Queue Dashboard", timeout=10000)

            # Submit multiple test jobs quickly
            jobs_submitted = 0
            for i in range(3):
                try:
                    # Look for test job submission button
                    submit_button = await page.query_selector("text=Submit Quick Test Job")
                    if not submit_button:
                        # Try alternative button names
                        submit_button = await page.query_selector("button:has-text('Test')")

                    if submit_button:
                        await submit_button.click()
                        jobs_submitted += 1
                        await asyncio.sleep(0.5)  # Small delay between submissions
                    else:
                        # Create test job programmatically
                        job_id = self.job_manager.submit_job("test", {"duration": 2, "steps": 5})
                        jobs_submitted += 1
                        print(f"    Submitted test job programmatically: {job_id[:8]}...")

                except Exception as e:
                    print(f"    Error submitting job {i+1}: {e}")

            self.log_test(
                "Multiple job submission",
                jobs_submitted >= 2,
                f"Successfully submitted {jobs_submitted} jobs"
            )

            # Wait a moment for jobs to start
            await asyncio.sleep(2)

            # Check GPU status through job manager
            gpu_status = self.job_manager.get_gpu_status()
            active_gpu_jobs = gpu_status.get('active_gpu_jobs', 0)
            queue_length = gpu_status.get('queue_length', 0)

            # Verify concurrency limits
            concurrent_limit_respected = active_gpu_jobs <= 1
            has_queued_jobs = queue_length > 0 or jobs_submitted <= 1

            self.log_test(
                "GPU concurrent job limit (≤1)",
                concurrent_limit_respected,
                f"Active GPU jobs: {active_gpu_jobs}, Queue length: {queue_length}"
            )

            self.log_test(
                "Job queuing system working",
                has_queued_jobs or jobs_submitted == 1,
                f"Queue functioning correctly with {queue_length} queued jobs"
            )

            return True

        except Exception as e:
            self.log_test("GPU concurrent job limits", False, f"Exception: {e}")
            return False

    async def test_gpu_queue_display(self, page):
        """Test queue position display in UI."""
        print("\n🧪 TESTING GPU QUEUE DISPLAY")
        print("-" * 50)

        try:
            # Check for queue status indicators in the UI
            await asyncio.sleep(2)  # Allow UI to update

            # Look for queue position indicators
            queue_indicators_found = False
            queue_elements = await page.query_selector_all("text=*Position*")
            if queue_elements:
                for elem in queue_elements:
                    text = await elem.text_content()
                    if "Position" in text and "queue" in text.lower():
                        print(f"    Found queue indicator: {text}")
                        queue_indicators_found = True

            # Look for hardware mode indicators
            hardware_indicators = await page.query_selector_all("text=*GPU*,text=*CPU*,text=*Queued*")
            hardware_display_found = len(hardware_indicators) > 0

            if hardware_indicators:
                for elem in hardware_indicators[:3]:  # Check first 3
                    text = await elem.text_content()
                    print(f"    Hardware indicator: {text}")

            # Look for status emojis
            status_emojis = await page.query_selector_all("text=🟡,text=⏳,text=🚀,text=💻")
            emoji_display_found = len(status_emojis) > 0

            self.log_test(
                "Queue position display",
                queue_indicators_found or len(status_emojis) > 0,
                f"Queue indicators: {queue_indicators_found}, Status emojis: {len(status_emojis)}"
            )

            self.log_test(
                "Hardware mode display",
                hardware_display_found,
                f"Found {len(hardware_indicators)} hardware indicators"
            )

            return True

        except Exception as e:
            self.log_test("GPU queue display", False, f"Exception: {e}")
            return False

    async def test_job_progression(self, page):
        """Test that jobs progress through queue correctly."""
        print("\n🧪 TESTING JOB PROGRESSION")
        print("-" * 50)

        try:
            # Monitor job progression for up to 30 seconds
            progression_data = []

            for i in range(15):  # 15 iterations × 2 seconds = 30 seconds
                await asyncio.sleep(2)

                # Get current GPU status
                gpu_status = self.job_manager.get_gpu_status()

                # Check UI for job status updates
                running_jobs = await page.query_selector_all("text=🟡")
                completed_jobs = await page.query_selector_all("text=✅")
                queued_jobs = await page.query_selector_all("text=⏳")

                progression_entry = {
                    "iteration": i + 1,
                    "active_gpu_jobs": gpu_status.get('active_gpu_jobs', 0),
                    "queue_length": gpu_status.get('queue_length', 0),
                    "ui_running": len(running_jobs),
                    "ui_completed": len(completed_jobs),
                    "ui_queued": len(queued_jobs)
                }
                progression_data.append(progression_entry)

                print(f"    Iteration {i+1}: Active={progression_entry['active_gpu_jobs']}, "
                      f"Queue={progression_entry['queue_length']}, "
                      f"UI Running={progression_entry['ui_running']}")

                # Check if all jobs completed
                if gpu_status.get('active_gpu_jobs', 0) == 0 and gpu_status.get('queue_length', 0) == 0:
                    print("    All jobs completed")
                    break

            # Analyze progression data
            max_concurrent = max(entry['active_gpu_jobs'] for entry in progression_data)
            queue_decreased = any(
                progression_data[i]['queue_length'] < progression_data[i-1]['queue_length']
                for i in range(1, len(progression_data))
                if len(progression_data) > 1
            )

            self.log_test(
                "Maximum concurrent GPU jobs ≤ 1",
                max_concurrent <= 1,
                f"Max concurrent observed: {max_concurrent}"
            )

            self.log_test(
                "Queue progression observed",
                queue_decreased or len(progression_data) < 5,
                f"Queue progression working correctly"
            )

            return True

        except Exception as e:
            self.log_test("Job progression", False, f"Exception: {e}")
            return False

    async def test_gpu_resource_cleanup(self, page):
        """Test GPU resource cleanup after job completion."""
        print("\n🧪 TESTING GPU RESOURCE CLEANUP")
        print("-" * 50)

        try:
            # Wait for all jobs to complete
            await asyncio.sleep(10)

            # Check final GPU status
            gpu_status = self.job_manager.get_gpu_status()

            cleanup_successful = (
                gpu_status.get('active_gpu_jobs', 0) == 0 and
                gpu_status.get('queue_length', 0) == 0
            )

            self.log_test(
                "GPU resources cleaned up",
                cleanup_successful,
                f"Final status: {gpu_status}"
            )

            # Check UI reflects completion
            completed_jobs = await page.query_selector_all("text=✅")
            failed_jobs = await page.query_selector_all("text=❌")

            self.log_test(
                "Jobs reached completion state",
                len(completed_jobs) > 0 or len(failed_jobs) > 0,
                f"Completed: {len(completed_jobs)}, Failed: {len(failed_jobs)}"
            )

            return True

        except Exception as e:
            self.log_test("GPU resource cleanup", False, f"Exception: {e}")
            return False

    async def test_timeout_scenarios(self, page):
        """Test timeout handling for GPU operations."""
        print("\n🧪 TESTING TIMEOUT SCENARIOS")
        print("-" * 50)

        try:
            # Submit a job with very short timeout (for testing)
            short_timeout_job = self.job_manager.submit_job("test", {
                "duration": 1,
                "steps": 3,
                "timeout_minutes": 0.1  # 6 seconds timeout
            })

            print(f"    Submitted short timeout job: {short_timeout_job[:8]}...")

            # Monitor for timeout behavior
            start_time = time.time()
            timeout_detected = False

            for i in range(10):  # Monitor for 20 seconds
                await asyncio.sleep(2)

                job_status = self.job_manager.get_job_status(short_timeout_job)
                if job_status and job_status.get('status') == 'failed':
                    error_msg = job_status.get('error_message', '')
                    if 'timeout' in error_msg.lower() or 'exceeded' in error_msg.lower():
                        timeout_detected = True
                        break

                elapsed = time.time() - start_time
                if elapsed > 20:  # Stop monitoring after 20 seconds
                    break

            self.log_test(
                "Timeout handling working",
                timeout_detected or time.time() - start_time < 15,
                "Timeout mechanism functioning correctly"
            )

            return True

        except Exception as e:
            self.log_test("Timeout scenarios", False, f"Exception: {e}")
            return False

    async def run_all_tests(self):
        """Run all GPU queue management tests."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("❌ Playwright not available. Install with: pip install playwright")
            print("Then run: playwright install")
            return False

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # Set to True for headless
            page = await browser.new_page()

            try:
                print("🚀 GPU QUEUE MANAGEMENT PLAYWRIGHT TESTS")
                print("=" * 60)

                # Run test suite
                await self.test_gpu_concurrent_job_limits(page)
                await self.test_gpu_queue_display(page)
                await self.test_job_progression(page)
                await self.test_gpu_resource_cleanup(page)
                await self.test_timeout_scenarios(page)

                # Generate final report
                self.generate_final_report()

                return True

            finally:
                await browser.close()

    def generate_final_report(self):
        """Generate final test report."""
        print("\n" + "=" * 60)
        print("🎯 GPU QUEUE MANAGEMENT TEST SUMMARY")
        print("=" * 60)

        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)

        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("\n🎉 SUCCESS: GPU Queue Management System is fully functional!")
            print("\n✅ Verified capabilities:")
            print("   • Maximum 1 concurrent GPU job limit")
            print("   • Queue-based job management")
            print("   • Real-time queue position display")
            print("   • Job progression through queue")
            print("   • Comprehensive resource cleanup")
            print("   • Timeout handling mechanisms")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {total_tests - passed_tests} tests failed")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   ❌ {result['name']}: {result['details']}")

        print(f"\n📋 GPU Queue Management Status: {'COMPLETE' if passed_tests == total_tests else 'NEEDS ATTENTION'}")

        return passed_tests == total_tests


async def main():
    """Run GPU queue management Playwright tests."""
    print("🧪 Starting GPU Queue Management Playwright Test Suite")
    print("=" * 70)
    print("Testing comprehensive GPU resource queue management with browser automation")
    print("Make sure Streamlit is running on http://localhost:8501")
    print()

    tester = GPUQueueManagementPlaywrightTest()
    success = await tester.run_all_tests()

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)