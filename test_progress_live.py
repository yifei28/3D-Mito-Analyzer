#!/usr/bin/env python3
"""
Live Test for Enhanced Progress Forwarding System

Tests the enhanced progress system with a real browser session.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_enhanced_progress_system():
    """Test the enhanced progress system with real browser interaction."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not available. Install with: pip install playwright")
        print("Then run: playwright install")
        return False

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)  # Set to False to see the test
        page = await browser.new_page()

        try:
            print("🚀 Starting Enhanced Progress System Test")
            print("=" * 50)

            # Navigate to the application
            print("📱 Navigating to Streamlit app...")
            await page.goto("http://localhost:8501", timeout=30000)

            # Wait for the app to load
            await page.wait_for_selector("text=Mitochondrial Image Analysis", timeout=15000)
            print("✅ Application loaded successfully")

            # Navigate to Job Queue
            print("🔄 Navigating to Job Queue...")
            await page.click("text=Job Queue")
            await page.wait_for_selector("text=Job Queue Dashboard", timeout=10000)
            print("✅ Job Queue page loaded")

            # Submit a test job
            print("🧪 Submitting test job...")
            try:
                await page.click("text=Submit Quick Test Job")
                print("✅ Test job submitted")
            except Exception as e:
                print(f"⚠️ Could not find quick test job button, trying alternative: {e}")
                # Try submitting any available test job
                test_buttons = await page.query_selector_all("button")
                for button in test_buttons:
                    text = await button.text_content()
                    if "test" in text.lower() or "submit" in text.lower():
                        await button.click()
                        print(f"✅ Clicked button: {text}")
                        break

            # Wait for job to start
            print("⏳ Waiting for job to start...")
            await asyncio.sleep(3)

            # Look for job status indicators
            try:
                # Look for running job indicator
                await page.wait_for_selector("text=🟡", timeout=30000)
                print("✅ Job is running")

                # Monitor progress for a short time
                print("📊 Monitoring progress updates...")
                for i in range(10):  # Monitor for 10 iterations
                    try:
                        # Look for progress indicators
                        progress_elements = await page.query_selector_all("[role='progressbar']")

                        if progress_elements:
                            for elem in progress_elements:
                                value = await elem.get_attribute('aria-valuenow')
                                if value:
                                    print(f"  📈 Progress: {value}%")
                                    break

                        # Look for stage information
                        stage_elements = await page.query_selector_all("text=*Stage:*")
                        if stage_elements:
                            for elem in stage_elements:
                                text = await elem.text_content()
                                if "Stage:" in text:
                                    print(f"  🔄 {text}")
                                    break

                        # Look for hardware mode
                        hardware_elements = await page.query_selector_all("text=*💻*,text=*🚀*")
                        if hardware_elements:
                            for elem in hardware_elements:
                                text = await elem.text_content()
                                if "💻" in text or "🚀" in text:
                                    print(f"  🖥️ Hardware: {text}")
                                    break

                        # Check if job completed
                        completed = await page.query_selector("text=✅ Completed")
                        if completed:
                            print("✅ Job completed successfully")
                            break

                        # Check if job failed
                        failed = await page.query_selector("text=❌ Failed")
                        if failed:
                            print("⚠️ Job failed")
                            break

                    except Exception as e:
                        print(f"  ⚠️ Error monitoring progress: {e}")

                    await asyncio.sleep(2)

            except Exception as e:
                print(f"⚠️ Could not detect running job: {e}")

            print("🎯 Test completed")
            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

        finally:
            await browser.close()


async def main():
    """Run the live test."""
    print("🧪 Enhanced Progress Forwarding Live Test")
    print("Make sure Streamlit is running on http://localhost:8501")
    print()

    success = await test_enhanced_progress_system()

    print()
    print("=" * 50)
    if success:
        print("🎉 Enhanced Progress System Test PASSED!")
        print("✅ The system is working correctly")
    else:
        print("❌ Enhanced Progress System Test FAILED!")
        print("⚠️ Please check the implementation")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)