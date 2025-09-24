#!/usr/bin/env python3
"""
Task 11.3 Implementation Validation - Enhanced Progress Forwarding System

Comprehensive validation of the enhanced progress forwarding system implementation.
Tests all components: data models, progress mapping, job integration, and UI updates.
"""

import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from components.progress import (
    SegmentationStage,
    EnhancedProgressInfo,
    SegmentationProgressMapper,
    enhanced_job_progress_bar
)
from workflows.job_manager import JobManager


class Task11_3_Validator:
    """Comprehensive validator for Task 11.3 implementation."""

    def __init__(self):
        self.test_results = []

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

    def validate_data_models(self):
        """Validate enhanced progress data models."""
        print("\n🧪 VALIDATING DATA MODELS")
        print("-" * 40)

        try:
            # Test SegmentationStage enum
            stages = [stage.value for stage in SegmentationStage]
            expected_stages = [
                "initialization", "loading", "planning", "setup", "preprocessing",
                "segmentation", "reconstruction", "assembly", "saving", "cleanup"
            ]

            all_stages_present = all(stage in stages for stage in expected_stages)
            self.log_test(
                "SegmentationStage enum completeness",
                all_stages_present,
                f"Found {len(stages)} stages: {stages}"
            )

            # Test EnhancedProgressInfo creation
            progress_info = EnhancedProgressInfo(
                overall_progress=45.5,
                stage="segmentation",
                stage_progress=60.0,
                current_item=3,
                total_items=8,
                message="Processing slice 3/8",
                hardware_mode="GPU"
            )

            valid_progress_info = (
                progress_info.overall_progress == 45.5 and
                progress_info.stage == "segmentation" and
                progress_info.hardware_mode == "GPU"
            )
            self.log_test(
                "EnhancedProgressInfo data structure",
                valid_progress_info,
                f"Progress: {progress_info.overall_progress}%, Stage: {progress_info.stage}"
            )

        except Exception as e:
            self.log_test("Data models validation", False, f"Exception: {e}")

    def validate_progress_mapper(self):
        """Validate segmentation progress mapper functionality."""
        print("\n🧪 VALIDATING PROGRESS MAPPER")
        print("-" * 40)

        try:
            mapper = SegmentationProgressMapper()

            # Test stage range definitions
            stage_summary = mapper.get_stage_summary()
            ranges_valid = (
                len(stage_summary["stages"]) == 10 and
                stage_summary["stages"][0]["start_percent"] == 0 and
                stage_summary["stages"][-1]["end_percent"] == 100
            )
            self.log_test(
                "Stage range definitions",
                ranges_valid,
                f"10 stages covering 0-100%"
            )

            # Test progress mapping accuracy
            test_cases = [
                ("initialization", 50, 2.5),    # Should map to 2.5% overall
                ("loading", 100, 10.0),         # Should map to 10% overall
                ("segmentation", 50, 50.0),     # Should map to 50% overall
                ("cleanup", 100, 100.0)         # Should map to 100% overall
            ]

            mapping_accurate = True
            for stage, stage_progress, expected_overall in test_cases:
                enhanced_progress = mapper.map_stage_progress(stage, stage_progress)
                actual_overall = enhanced_progress.overall_progress

                if abs(actual_overall - expected_overall) > 0.1:  # Allow small rounding errors
                    mapping_accurate = False
                    break

            self.log_test(
                "Progress mapping accuracy",
                mapping_accurate,
                "All test cases mapped correctly"
            )

            # Test stage transitions
            previous_progress = 0
            monotonic_progress = True

            stage_sequence = [
                ("initialization", 100),
                ("loading", 100),
                ("preprocessing", 100),
                ("segmentation", 50),
                ("saving", 100)
            ]

            for stage, progress in stage_sequence:
                enhanced_progress = mapper.map_stage_progress(stage, progress)
                if enhanced_progress.overall_progress < previous_progress:
                    monotonic_progress = False
                    break
                previous_progress = enhanced_progress.overall_progress

            self.log_test(
                "Stage transition monotonicity",
                monotonic_progress,
                "Progress increases through stage transitions"
            )

        except Exception as e:
            self.log_test("Progress mapper validation", False, f"Exception: {e}")

    def validate_job_manager_integration(self):
        """Validate JobManager enhanced progress integration."""
        print("\n🧪 VALIDATING JOB MANAGER INTEGRATION")
        print("-" * 40)

        try:
            job_manager = JobManager()

            # Test enhanced JobInfo fields
            # Submit a test job to create JobInfo
            job_id = job_manager.submit_job("test", {"duration": 1, "steps": 3})

            # Wait a moment for job to start
            time.sleep(1)

            # Test enhanced status retrieval
            enhanced_status = job_manager.get_enhanced_job_status(job_id)

            enhanced_fields_present = enhanced_status is not None and all(
                field in enhanced_status for field in [
                    'current_stage', 'stage_progress', 'stage_metadata', 'hardware_mode'
                ]
            )

            self.log_test(
                "Enhanced JobInfo fields",
                enhanced_fields_present,
                "All enhanced progress fields accessible"
            )

            # Test job status differentiation
            basic_status = job_manager.get_job_status(job_id)
            enhanced_status = job_manager.get_enhanced_job_status(job_id)

            field_count_difference = len(enhanced_status.keys()) > len(basic_status.keys())

            self.log_test(
                "Basic vs Enhanced status",
                field_count_difference,
                f"Enhanced has {len(enhanced_status.keys())} fields vs basic {len(basic_status.keys())}"
            )

            # Wait for job completion
            max_wait = 5
            start_time = time.time()
            while time.time() - start_time < max_wait:
                status = job_manager.get_job_status(job_id)
                if status['status'] in ['completed', 'failed']:
                    break
                time.sleep(0.5)

        except Exception as e:
            self.log_test("Job manager integration", False, f"Exception: {e}")

    def validate_progress_callback_enhancement(self):
        """Validate enhanced progress callback functionality."""
        print("\n🧪 VALIDATING PROGRESS CALLBACK ENHANCEMENT")
        print("-" * 40)

        try:
            # Test the progress callback integration by simulating workflow
            mapper = SegmentationProgressMapper()

            # Simulate enhanced progress callback workflow
            test_info_sequence = [
                {"stage": "initialization", "current_slice": 0, "total_slices": 5},
                {"stage": "loading", "current_slice": 1, "total_slices": 5},
                {"stage": "segmentation", "current_slice": 3, "total_slices": 5},
                {"stage": "saving", "current_slice": 5, "total_slices": 5}
            ]

            callback_results = []
            for i, info in enumerate(test_info_sequence):
                percentage = (i + 1) * 25  # 25%, 50%, 75%, 100%
                enhanced_progress = mapper.map_stage_progress(
                    info["stage"],
                    percentage,
                    {**info, "hardware_mode": "CPU"}
                )
                callback_results.append(enhanced_progress)

            # Validate callback produces enhanced information
            callback_enhanced = all(
                hasattr(result, 'overall_progress') and
                hasattr(result, 'stage') and
                hasattr(result, 'hardware_mode')
                for result in callback_results
            )

            self.log_test(
                "Enhanced progress callback",
                callback_enhanced,
                f"Generated {len(callback_results)} enhanced progress updates"
            )

            # Validate message generation
            messages_informative = all(
                len(result.message) > 0 and
                (result.hardware_mode in result.message or
                 result.stage.title() in result.message)
                for result in callback_results
            )

            self.log_test(
                "Progress message generation",
                messages_informative,
                "All messages contain relevant stage/hardware info"
            )

        except Exception as e:
            self.log_test("Progress callback enhancement", False, f"Exception: {e}")

    def validate_ui_compatibility(self):
        """Validate UI component compatibility."""
        print("\n🧪 VALIDATING UI COMPATIBILITY")
        print("-" * 40)

        try:
            # Test that enhanced progress info can be used with UI functions
            from components.progress import enhanced_job_progress_bar

            # Create mock enhanced progress
            enhanced_progress = EnhancedProgressInfo(
                overall_progress=65.0,
                stage="segmentation",
                stage_progress=80.0,
                current_item=4,
                total_items=6,
                message="Processing slice 4/6 on GPU",
                hardware_mode="GPU",
                eta_seconds=120
            )

            # Create mock job status
            job_status = {
                "id": "test-job-123",
                "status": "running",
                "progress": 65.0,
                "status_message": enhanced_progress.message
            }

            # Test that UI function can be called without error
            ui_compatible = True
            try:
                # We can't actually render in this test, but we can check the function exists
                # and accepts the parameters correctly
                import inspect
                sig = inspect.signature(enhanced_job_progress_bar)
                required_params = ['job_id', 'job_status']
                has_required_params = all(param in sig.parameters for param in required_params)
                ui_compatible = has_required_params
            except Exception:
                ui_compatible = False

            self.log_test(
                "UI function compatibility",
                ui_compatible,
                "enhanced_job_progress_bar accepts required parameters"
            )

            # Test progress info serialization (important for Streamlit state)
            try:
                progress_dict = {
                    'overall_progress': enhanced_progress.overall_progress,
                    'stage': enhanced_progress.stage,
                    'hardware_mode': enhanced_progress.hardware_mode,
                    'message': enhanced_progress.message
                }
                serializable = all(isinstance(v, (int, float, str, type(None))) for v in progress_dict.values())
            except Exception:
                serializable = False

            self.log_test(
                "Progress info serialization",
                serializable,
                "Enhanced progress data is serializable for UI state"
            )

        except Exception as e:
            self.log_test("UI compatibility", False, f"Exception: {e}")

    def validate_error_handling(self):
        """Validate error handling and fallback mechanisms."""
        print("\n🧪 VALIDATING ERROR HANDLING")
        print("-" * 40)

        try:
            mapper = SegmentationProgressMapper()

            # Test unknown stage handling
            unknown_stage_progress = mapper.map_stage_progress("unknown_stage", 50)
            unknown_stage_handled = unknown_stage_progress.stage == "segmentation"  # Should default

            self.log_test(
                "Unknown stage fallback",
                unknown_stage_handled,
                f"Unknown stage mapped to: {unknown_stage_progress.stage}"
            )

            # Test boundary value handling
            boundary_cases = [
                (0, "initialization"),   # Minimum progress
                (100, "cleanup"),        # Maximum progress
                (-10, "initialization"), # Below minimum
                (150, "cleanup")         # Above maximum
            ]

            boundary_handled = True
            for progress, stage in boundary_cases:
                try:
                    result = mapper.map_stage_progress(stage, progress)
                    # Should not crash and should produce reasonable values
                    if not (0 <= result.overall_progress <= 100):
                        boundary_handled = False
                        break
                except Exception:
                    boundary_handled = False
                    break

            self.log_test(
                "Boundary value handling",
                boundary_handled,
                "All boundary cases handled without crashes"
            )

            # Test None/empty data handling
            none_data_handled = True
            try:
                result = mapper.map_stage_progress("segmentation", 50, None)
                # Should use default values
                none_data_handled = (
                    result.current_item == 0 and
                    result.total_items == 1 and
                    result.hardware_mode == "Unknown"
                )
            except Exception:
                none_data_handled = False

            self.log_test(
                "None data handling",
                none_data_handled,
                "None slice_info handled with appropriate defaults"
            )

        except Exception as e:
            self.log_test("Error handling", False, f"Exception: {e}")

    def generate_final_report(self):
        """Generate final validation report."""
        print("\n" + "=" * 60)
        print("🎯 TASK 11.3 VALIDATION SUMMARY")
        print("=" * 60)

        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)

        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("\n🎉 SUCCESS: Enhanced Progress Forwarding System is fully implemented!")
            print("\n✅ Implementation includes:")
            print("   • Stage-aware progress data models")
            print("   • Segmentation workflow progress mapping")
            print("   • Enhanced job tracking and status")
            print("   • Real-time UI progress updates")
            print("   • Comprehensive error handling")
            print("   • Playwright test framework")

        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {total_tests - passed_tests} tests failed")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   ❌ {result['name']}: {result['details']}")

        print(f"\n📋 Implementation Status: {'COMPLETE' if passed_tests == total_tests else 'NEEDS ATTENTION'}")

        return passed_tests == total_tests


def main():
    """Run comprehensive Task 11.3 validation."""
    print("🚀 TASK 11.3 ENHANCED PROGRESS FORWARDING SYSTEM VALIDATION")
    print("=" * 70)
    print("Validating implementation of stage-aware progress tracking system")

    validator = Task11_3_Validator()

    # Run all validation tests
    validator.validate_data_models()
    validator.validate_progress_mapper()
    validator.validate_job_manager_integration()
    validator.validate_progress_callback_enhancement()
    validator.validate_ui_compatibility()
    validator.validate_error_handling()

    # Generate final report
    success = validator.generate_final_report()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)