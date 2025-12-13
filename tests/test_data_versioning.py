import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from neural.data import (
    DataQualityValidator,
    DatasetVersionManager,
    Feature,
    FeatureStore,
    LineageTracker,
    PreprocessingTracker,
)


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / ".neural_data_test"
    yield data_dir
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture
def sample_dataset(tmp_path):
    data = np.random.rand(100, 10)
    file_path = tmp_path / "sample_data.npy"
    np.save(file_path, data)
    yield file_path
    if file_path.exists():
        file_path.unlink()


def test_dataset_versioning(temp_data_dir, sample_dataset):
    manager = DatasetVersionManager(base_dir=temp_data_dir)
    
    version = manager.create_version(
        dataset_path=sample_dataset,
        metadata={"samples": 100, "features": 10},
        tags=["test", "v1"],
    )
    
    assert version.version is not None
    assert len(version.checksum) == 64
    assert "test" in version.tags
    assert version.metadata["samples"] == 100
    
    versions = manager.list_versions()
    assert len(versions) == 1
    
    retrieved = manager.get_version(version.version)
    assert retrieved is not None
    assert retrieved.checksum == version.checksum


def test_version_comparison(temp_data_dir, sample_dataset):
    manager = DatasetVersionManager(base_dir=temp_data_dir)
    
    version1 = manager.create_version(
        dataset_path=sample_dataset,
        version="v1.0",
        metadata={"samples": 100},
    )
    
    data2 = np.random.rand(200, 10)
    np.save(sample_dataset, data2)
    
    version2 = manager.create_version(
        dataset_path=sample_dataset,
        version="v2.0",
        metadata={"samples": 200},
    )
    
    comparison = manager.compare_versions("v1.0", "v2.0")
    
    assert comparison["checksum_match"] is False
    assert len(comparison["metadata_diff"]) > 0


def test_preprocessing_pipeline(temp_data_dir):
    tracker = PreprocessingTracker(base_dir=temp_data_dir)
    
    pipeline = tracker.create_pipeline(
        name="test_pipeline",
        description="Test preprocessing"
    )
    
    step1 = pipeline.add_step(
        name="normalize",
        params={"min": 0, "max": 1},
        description="Normalize data"
    )
    
    step2 = pipeline.add_step(
        name="scale",
        params={"factor": 2},
        description="Scale data"
    )
    
    tracker.update_pipeline(pipeline)
    
    retrieved = tracker.get_pipeline("test_pipeline")
    assert retrieved is not None
    assert len(retrieved.steps) == 2
    assert retrieved.steps[0].name == "normalize"
    
    pipelines = tracker.list_pipelines()
    assert "test_pipeline" in pipelines


def test_feature_store(temp_data_dir):
    store = FeatureStore(base_dir=temp_data_dir)
    
    group = store.create_feature_group(
        name="test_features",
        description="Test feature group"
    )
    
    feature1 = store.add_feature(
        "test_features",
        "feature_1",
        "float32",
        "First feature"
    )
    
    feature2 = store.add_feature(
        "test_features",
        "feature_2",
        "int32",
        "Second feature"
    )
    
    retrieved_group = store.get_feature_group("test_features")
    assert retrieved_group is not None
    assert len(retrieved_group.features) == 2
    
    stats = store.get_feature_statistics()
    assert stats["total_features"] == 2
    assert stats["total_groups"] == 1
    
    results = store.search_features("feature")
    assert len(results) == 2


def test_quality_validation():
    validator = DataQualityValidator(base_dir=".neural_data_test")
    
    clean_data = np.random.rand(100, 10)
    results = validator.validate(clean_data)
    
    assert len(results) > 0
    passed = sum(1 for r in results if r.passed)
    assert passed > 0
    
    data_with_nan = clean_data.copy()
    data_with_nan[0, 0] = np.nan
    
    results_with_nan = validator.validate(data_with_nan)
    
    no_missing_result = next(
        (r for r in results_with_nan if r.rule_name == "no_missing_values"),
        None
    )
    assert no_missing_result is not None
    assert no_missing_result.passed is False


def test_custom_validation_rule():
    validator = DataQualityValidator(base_dir=".neural_data_test")
    
    rule = validator.create_custom_rule(
        name="min_100_rows",
        rule_type="size",
        condition="min_rows",
        threshold=100,
        description="Must have at least 100 rows"
    )
    
    assert rule is not None
    
    small_data = np.random.rand(50, 10)
    results = validator.validate(small_data, rules=["min_100_rows"])
    
    assert len(results) == 1
    assert results[0].passed is False
    
    large_data = np.random.rand(150, 10)
    results = validator.validate(large_data, rules=["min_100_rows"])
    
    assert len(results) == 1
    assert results[0].passed is True


def test_lineage_tracking(temp_data_dir):
    tracker = LineageTracker(base_dir=temp_data_dir)
    
    graph = tracker.create_graph("test_experiment")
    
    data_node = tracker.add_data_node(
        "test_experiment",
        "data_1",
        "Training Data",
        {"samples": 1000}
    )
    
    model_node = tracker.add_model_node(
        "test_experiment",
        "model_1",
        "Neural Network",
        {"layers": 5}
    )
    
    pred_node = tracker.add_prediction_node(
        "test_experiment",
        "pred_1",
        "Predictions",
        {"accuracy": 0.95}
    )
    
    edge1 = tracker.add_edge(
        "test_experiment",
        data_node.node_id,
        model_node.node_id,
        "train"
    )
    
    edge2 = tracker.add_edge(
        "test_experiment",
        model_node.node_id,
        pred_node.node_id,
        "predict"
    )
    
    retrieved_graph = tracker.get_graph("test_experiment")
    assert retrieved_graph is not None
    assert len(retrieved_graph.nodes) == 3
    assert len(retrieved_graph.edges) == 2
    
    successors = retrieved_graph.get_successors(data_node.node_id)
    assert len(successors) == 1
    assert successors[0].node_id == model_node.node_id
    
    lineage = tracker.get_full_lineage("test_experiment", data_node.node_id)
    assert len(lineage["downstream"]) > 0


def test_lineage_path(temp_data_dir):
    tracker = LineageTracker(base_dir=temp_data_dir)
    graph = tracker.create_graph("path_test")
    
    node1 = tracker.add_data_node("path_test", "n1", "Node 1")
    node2 = tracker.add_preprocessing_node("path_test", "n2", "Node 2")
    node3 = tracker.add_model_node("path_test", "n3", "Node 3")
    
    tracker.add_edge("path_test", "n1", "n2", "transform")
    tracker.add_edge("path_test", "n2", "n3", "train")
    
    path = graph.get_path("n1", "n3")
    assert path is not None
    assert len(path) == 3
    assert path[0].node_id == "n1"
    assert path[-1].node_id == "n3"


def test_version_tagging(temp_data_dir, sample_dataset):
    manager = DatasetVersionManager(base_dir=temp_data_dir)
    
    version = manager.create_version(
        dataset_path=sample_dataset,
        tags=["initial"],
    )
    
    success = manager.tag_version(version.version, ["production", "v1.0"])
    assert success is True
    
    retrieved = manager.get_version(version.version)
    assert "production" in retrieved.tags
    assert "v1.0" in retrieved.tags
    assert "initial" in retrieved.tags


def test_version_deletion(temp_data_dir, sample_dataset):
    manager = DatasetVersionManager(base_dir=temp_data_dir)
    
    version = manager.create_version(dataset_path=sample_dataset)
    version_name = version.version
    
    assert manager.get_version(version_name) is not None
    
    success = manager.delete_version(version_name)
    assert success is True
    
    assert manager.get_version(version_name) is None


def test_pipeline_comparison(temp_data_dir):
    tracker = PreprocessingTracker(base_dir=temp_data_dir)
    
    pipeline1 = tracker.create_pipeline("pipeline1")
    pipeline1.add_step("normalize", params={"min": 0, "max": 1})
    pipeline1.add_step("scale", params={"factor": 2})
    tracker.update_pipeline(pipeline1)
    
    pipeline2 = tracker.create_pipeline("pipeline2")
    pipeline2.add_step("normalize", params={"min": 0, "max": 255})
    pipeline2.add_step("augment", params={"rotation": 15})
    tracker.update_pipeline(pipeline2)
    
    comparison = tracker.compare_pipelines("pipeline1", "pipeline2")
    
    assert comparison["total_steps1"] == 2
    assert comparison["total_steps2"] == 2
    assert len(comparison["differences"]) > 0


def test_feature_export_import(temp_data_dir, tmp_path):
    store = FeatureStore(base_dir=temp_data_dir)
    
    group = store.create_feature_group("export_test")
    store.add_feature("export_test", "f1", "float32", "Feature 1")
    store.add_feature("export_test", "f2", "int32", "Feature 2")
    
    export_path = tmp_path / "features.json"
    success = store.export_feature_group("export_test", export_path)
    assert success is True
    assert export_path.exists()
    
    store.delete_feature_group("export_test")
    
    imported_group = store.import_feature_group(export_path)
    assert imported_group.name == "export_test"
    assert len(imported_group.features) == 2


def test_validation_history(temp_data_dir):
    validator = DataQualityValidator(base_dir=temp_data_dir)
    
    data = np.random.rand(100, 10)
    
    results1 = validator.validate_and_save(data, "test_dataset")
    results2 = validator.validate_and_save(data, "test_dataset")
    
    history = validator.get_validation_history("test_dataset")
    assert len(history) >= 2
    
    report = validator.get_quality_report("test_dataset")
    assert report["dataset_name"] == "test_dataset"
    assert report["total_validations"] >= 2


def test_lineage_export_import(temp_data_dir, tmp_path):
    tracker = LineageTracker(base_dir=temp_data_dir)
    
    graph = tracker.create_graph("export_test")
    tracker.add_data_node("export_test", "d1", "Data 1")
    tracker.add_model_node("export_test", "m1", "Model 1")
    tracker.add_edge("export_test", "d1", "m1", "train")
    
    export_path = tmp_path / "lineage.json"
    success = tracker.export_graph("export_test", export_path)
    assert success is True
    assert export_path.exists()
    
    tracker.delete_graph("export_test")
    
    imported_graph = tracker.import_graph(export_path)
    assert imported_graph.name == "export_test"
    assert len(imported_graph.nodes) == 2
    assert len(imported_graph.edges) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
