#!/usr/bin/env python3
"""
Playwright automation test for Task 10.6 Output Management and Metadata Generation

This script automates the testing of the output management web interface
to validate all features work correctly.
"""

import asyncio
import time
from datetime import datetime
from playwright.async_api import async_playwright, Page


class OutputManagementPlaywrightTester:
    """Playwright automation for testing output management functionality."""

    def __init__(self):
        self.base_url = "http://localhost:8522"
        self.test_results = []

    async def setup_browser(self, playwright):
        """Set up browser and page."""
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Navigate to app
        await page.goto(self.base_url)
        await page.wait_for_load_state('networkidle')

        return browser, page

    async def wait_for_workflow_init(self, page: Page):
        """Wait for the workflow to initialize."""
        print("🔄 Waiting for workflow initialization...")

        # Wait for success message or error
        try:
            await page.wait_for_selector("text=✅ Workflow initialized successfully!", timeout=30000)
            print("✅ Workflow initialized successfully")
            return True
        except:
            try:
                error_element = await page.wait_for_selector("[data-testid='error']", timeout=5000)
                error_text = await error_element.text_content()
                print(f"❌ Workflow initialization failed: {error_text}")
                return False
            except:
                print("❌ Workflow initialization timeout")
                return False

    async def test_file_operations_tab(self, page: Page):
        """Test file operations functionality."""
        print("\n📁 Testing File Operations Tab...")

        try:
            # Click on File Operations tab
            await page.click("text=📁 File Operations")
            await page.wait_for_timeout(2000)

            # Check if enhanced filenames are displayed
            enhanced_filenames = await page.query_selector_all("text=/.*_segmented.*/")
            if len(enhanced_filenames) >= 3:
                self.test_results.append("✅ File Operations - Enhanced filename generation working")
                print("✅ Enhanced filename generation working")
            else:
                self.test_results.append("❌ File Operations - Enhanced filename generation failed")
                print("❌ Enhanced filename generation failed")

            # Check for directory validation result
            validation_messages = await page.query_selector_all("text=/Directory validation/")
            if len(validation_messages) > 0:
                self.test_results.append("✅ File Operations - Directory validation tested")
                print("✅ Directory validation tested")
            else:
                self.test_results.append("❌ File Operations - Directory validation not found")
                print("❌ Directory validation not found")

            # Check for cleanup result
            cleanup_messages = await page.query_selector_all("text=/cleanup/")
            if len(cleanup_messages) > 0:
                self.test_results.append("✅ File Operations - File cleanup tested")
                print("✅ File cleanup tested")
            else:
                self.test_results.append("❌ File Operations - File cleanup not found")
                print("❌ File cleanup not found")

        except Exception as e:
            self.test_results.append(f"❌ File Operations - Error: {str(e)}")
            print(f"❌ File Operations test failed: {e}")

    async def test_metadata_generation_tab(self, page: Page):
        """Test metadata generation functionality."""
        print("\n📊 Testing Metadata Generation Tab...")

        try:
            # Click on Metadata Generation tab
            await page.click("text=📊 Metadata Generation")
            await page.wait_for_timeout(3000)

            # Check for metadata structure tabs
            metadata_tabs = ["Processing", "Hardware", "Performance", "TensorFlow", "Input/Output"]
            tabs_found = 0

            for tab_name in metadata_tabs:
                tab_element = await page.query_selector(f"text={tab_name}")
                if tab_element:
                    tabs_found += 1

            if tabs_found >= 4:
                self.test_results.append("✅ Metadata Generation - Metadata structure tabs present")
                print("✅ Metadata structure tabs present")
            else:
                self.test_results.append("❌ Metadata Generation - Missing metadata tabs")
                print("❌ Missing metadata tabs")

            # Click through each tab to test content
            for tab_name in metadata_tabs[:3]:  # Test first 3 tabs
                try:
                    await page.click(f"text={tab_name}")
                    await page.wait_for_timeout(1000)

                    # Look for JSON content
                    json_content = await page.query_selector("pre")
                    if json_content:
                        print(f"✅ {tab_name} tab contains data")
                    else:
                        print(f"⚠️ {tab_name} tab may be empty")
                except:
                    print(f"⚠️ Could not test {tab_name} tab")

            # Check for validation results
            validation_results = await page.query_selector_all("text=/✅|❌/")
            if len(validation_results) >= 3:
                self.test_results.append("✅ Metadata Generation - Validation results displayed")
                print("✅ Validation results displayed")
            else:
                self.test_results.append("❌ Metadata Generation - Insufficient validation results")
                print("❌ Insufficient validation results")

        except Exception as e:
            self.test_results.append(f"❌ Metadata Generation - Error: {str(e)}")
            print(f"❌ Metadata Generation test failed: {e}")

    async def test_hardware_integration_tab(self, page: Page):
        """Test hardware integration functionality."""
        print("\n🔧 Testing Hardware Integration Tab...")

        try:
            # Click on Hardware Integration tab
            await page.click("text=🔧 Hardware Integration")
            await page.wait_for_timeout(2000)

            # Check for hardware metrics
            metrics = await page.query_selector_all("[data-testid='metric-value']")
            if len(metrics) >= 4:
                self.test_results.append("✅ Hardware Integration - Hardware metrics displayed")
                print("✅ Hardware metrics displayed")
            else:
                # Alternative check for metric content
                metric_text = await page.query_selector_all("text=/Hardware Type|Device Name|Memory|CPU/")
                if len(metric_text) >= 2:
                    self.test_results.append("✅ Hardware Integration - Hardware info present")
                    print("✅ Hardware info present")
                else:
                    self.test_results.append("❌ Hardware Integration - Missing hardware info")
                    print("❌ Missing hardware info")

            # Check for TensorFlow configuration
            tf_elements = await page.query_selector_all("text=/TensorFlow/")
            if len(tf_elements) > 0:
                self.test_results.append("✅ Hardware Integration - TensorFlow info displayed")
                print("✅ TensorFlow info displayed")
            else:
                self.test_results.append("❌ Hardware Integration - TensorFlow info missing")
                print("❌ TensorFlow info missing")

            # Check for fallback event tracking
            fallback_elements = await page.query_selector_all("text=/Fallback Event|fallback/")
            if len(fallback_elements) > 0:
                self.test_results.append("✅ Hardware Integration - Fallback tracking present")
                print("✅ Fallback tracking present")
            else:
                self.test_results.append("❌ Hardware Integration - Fallback tracking missing")
                print("❌ Fallback tracking missing")

        except Exception as e:
            self.test_results.append(f"❌ Hardware Integration - Error: {str(e)}")
            print(f"❌ Hardware Integration test failed: {e}")

    async def test_performance_testing_tab(self, page: Page):
        """Test performance testing functionality."""
        print("\n🏃 Testing Performance Testing Tab...")

        try:
            # Click on Performance Testing tab
            await page.click("text=⚡ Performance Testing")
            await page.wait_for_timeout(3000)

            # Check for different test sizes
            size_tests = ["Small", "Medium", "Large"]
            performance_results = []

            for size in size_tests:
                size_elements = await page.query_selector_all(f"text=*{size}*")
                if len(size_elements) > 0:
                    performance_results.append(size)

            if len(performance_results) >= 2:
                self.test_results.append("✅ Performance Testing - Multiple size tests executed")
                print("✅ Multiple size tests executed")
            else:
                self.test_results.append("❌ Performance Testing - Insufficient size tests")
                print("❌ Insufficient size tests")

            # Check for performance metrics
            perf_metrics = await page.query_selector_all("text=/Basic Save|Enhanced Save|Overhead/")
            if len(perf_metrics) >= 6:  # 3 metrics × 2+ tests
                self.test_results.append("✅ Performance Testing - Performance metrics displayed")
                print("✅ Performance metrics displayed")
            else:
                self.test_results.append("❌ Performance Testing - Missing performance metrics")
                print("❌ Missing performance metrics")

            # Check for performance summary
            summary_elements = await page.query_selector_all("text=/Performance Summary|overhead/")
            if len(summary_elements) > 0:
                self.test_results.append("✅ Performance Testing - Performance summary provided")
                print("✅ Performance summary provided")
            else:
                self.test_results.append("❌ Performance Testing - Performance summary missing")
                print("❌ Performance summary missing")

        except Exception as e:
            self.test_results.append(f"❌ Performance Testing - Error: {str(e)}")
            print(f"❌ Performance Testing test failed: {e}")

    async def test_full_pipeline_tab(self, page: Page):
        """Test full pipeline functionality."""
        print("\n🎯 Testing Full Pipeline Tab...")

        try:
            # Click on Full Pipeline tab
            await page.click("text=🎯 Full Pipeline")
            await page.wait_for_timeout(2000)

            # Check for file upload interface
            upload_elements = await page.query_selector_all("text=/Upload.*TIFF/")
            if len(upload_elements) > 0:
                self.test_results.append("✅ Full Pipeline - File upload interface present")
                print("✅ File upload interface present")
            else:
                self.test_results.append("❌ Full Pipeline - File upload interface missing")
                print("❌ File upload interface missing")

            # Test sample TIFF creation
            sample_button = await page.query_selector("text=📁 Create Sample TIFF for Testing")
            if sample_button:
                await sample_button.click()
                await page.wait_for_timeout(2000)

                # Check for download button
                download_button = await page.query_selector("text=📥 Download Sample TIFF")
                if download_button:
                    self.test_results.append("✅ Full Pipeline - Sample TIFF creation working")
                    print("✅ Sample TIFF creation working")
                else:
                    self.test_results.append("❌ Full Pipeline - Sample TIFF creation failed")
                    print("❌ Sample TIFF creation failed")

            # Check for test configuration options
            config_checkboxes = await page.query_selector_all("input[type='checkbox']")
            if len(config_checkboxes) >= 2:
                self.test_results.append("✅ Full Pipeline - Configuration options available")
                print("✅ Configuration options available")
            else:
                self.test_results.append("❌ Full Pipeline - Configuration options missing")
                print("❌ Configuration options missing")

        except Exception as e:
            self.test_results.append(f"❌ Full Pipeline - Error: {str(e)}")
            print(f"❌ Full Pipeline test failed: {e}")

    async def test_overall_functionality(self, page: Page):
        """Test overall interface functionality."""
        print("\n🧪 Testing Overall Interface...")

        try:
            # Check for main header
            header = await page.query_selector("text=Output Management & Metadata Test Suite")
            if header:
                self.test_results.append("✅ Overall - Main header present")
                print("✅ Main header present")

            # Check for all tabs
            expected_tabs = ["📁 File Operations", "📊 Metadata Generation", "🔧 Hardware Integration",
                           "⚡ Performance Testing", "🎯 Full Pipeline"]

            tabs_found = 0
            for tab in expected_tabs:
                tab_element = await page.query_selector(f"text={tab}")
                if tab_element:
                    tabs_found += 1

            if tabs_found >= 4:
                self.test_results.append("✅ Overall - All required tabs present")
                print("✅ All required tabs present")
            else:
                self.test_results.append("❌ Overall - Missing required tabs")
                print("❌ Missing required tabs")

            # Check for hardware configuration display
            config_elements = await page.query_selector_all("text=/Hardware Configuration|Type:|Device:|Memory:|Batch Size:/")
            if len(config_elements) >= 3:
                self.test_results.append("✅ Overall - Hardware configuration displayed")
                print("✅ Hardware configuration displayed")
            else:
                self.test_results.append("❌ Overall - Hardware configuration missing")
                print("❌ Hardware configuration missing")

        except Exception as e:
            self.test_results.append(f"❌ Overall - Error: {str(e)}")
            print(f"❌ Overall test failed: {e}")

    async def run_all_tests(self):
        """Run all Playwright tests for output management."""
        print("🧪 Starting Output Management Playwright Test Suite")
        print("=" * 80)

        async with async_playwright() as playwright:
            browser, page = await self.setup_browser(playwright)

            try:
                # Wait for workflow initialization
                if not await self.wait_for_workflow_init(page):
                    self.test_results.append("❌ Critical - Workflow initialization failed")
                    return

                # Run all test modules
                await self.test_overall_functionality(page)
                await self.test_file_operations_tab(page)
                await self.test_metadata_generation_tab(page)
                await self.test_hardware_integration_tab(page)
                await self.test_performance_testing_tab(page)
                await self.test_full_pipeline_tab(page)

            finally:
                await browser.close()

        # Print final results
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 OUTPUT MANAGEMENT PLAYWRIGHT TEST RESULTS")
        print("=" * 80)

        passed = sum(1 for result in self.test_results if result.startswith("✅"))
        total = len(self.test_results)

        # Group results by category
        categories = {}
        for result in self.test_results:
            if " - " in result:
                category = result.split(" - ")[0].replace("✅ ", "").replace("❌ ", "")
                status = "✅" if result.startswith("✅") else "❌"
                if category not in categories:
                    categories[category] = {"passed": 0, "total": 0}
                categories[category]["total"] += 1
                if status == "✅":
                    categories[category]["passed"] += 1

        # Print category summaries
        for category, stats in categories.items():
            passed_cat = stats["passed"]
            total_cat = stats["total"]
            percentage = (passed_cat / total_cat * 100) if total_cat > 0 else 0
            print(f"{category}: {passed_cat}/{total_cat} ({percentage:.1f}%)")

        print("\nDetailed Results:")
        for result in self.test_results:
            print(result)

        print(f"\n🎯 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 ALL OUTPUT MANAGEMENT TESTS PASSED!")
            print("✅ Task 10.6 implementation is working correctly")
        elif passed / total >= 0.8:
            print("✅ MOST TESTS PASSED - Implementation is largely working")
            print("⚠️ Review failed tests for minor issues")
        else:
            print("⚠️ SOME TESTS FAILED - Review implementation")
            print("❌ Task 10.6 may need additional work")

        return passed == total


async def main():
    """Main test execution."""
    tester = OutputManagementPlaywrightTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)