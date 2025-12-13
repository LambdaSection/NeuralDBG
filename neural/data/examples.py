from pathlib import Path

import numpy as np

from .dataset_version import DatasetVersionManager
from .dvc_integration import DVCIntegration
from .feature_store import Feature, FeatureStore
from .lineage_tracker import LineageTracker
from .preprocessing_tracker import PreprocessingTracker
from .quality_validator import DataQualityValidator
from .utils import create_data_to_model_lineage


def example_dataset_versioning():
    print("=" * 60)
    print("Dataset Versioning Example")
    print("=" * 60)
    
    base_dir = Path(".neural_data_example")
    manager = DatasetVersionManager(base_dir=base_dir)
    
    temp_data = np.random.rand(100, 10)
    temp_file = Path("temp_dataset.npy")
    np.save(temp_file, temp_data)
    
    version1 = manager.create_version(
        dataset_path=temp_file,
        metadata={"samples": 100, "features": 10},
        tags=["training", "v1"],
    )
    
    print(f"\nCreated version: {version1.version}")
    print(f"Checksum: {version1.checksum[:16]}...")
    print(f"Tags: {version1.tags}")
    
    temp_data_v2 = np.random.rand(200, 10)
    np.save(temp_file, temp_data_v2)
    
    version2 = manager.create_version(
        dataset_path=temp_file,
        version="v2.0",
        metadata={"samples": 200, "features": 10},
        tags=["training", "v2"],
    )
    
    print(f"\nCreated version: {version2.version}")
    print(f"Checksum: {version2.checksum[:16]}...")
    
    versions = manager.list_versions(tags=["training"])
    print(f"\nTotal versions with 'training' tag: {len(versions)}")
    
    comparison = manager.compare_versions(version1.version, version2.version)
    print(f"\nChecksum match: {comparison['checksum_match']}")
    print(f"Metadata differences: {len(comparison['metadata_diff'])}")
    
    temp_file.unlink()
    
    print("\n✓ Dataset versioning example completed!")


def example_dvc_integration():
    print("\n" + "=" * 60)
    print("DVC Integration Example")
    print("=" * 60)
    
    dvc = DVCIntegration()
    
    if not dvc.is_dvc_available():
        print("\nDVC is not installed. Install with: pip install dvc")
        return
    
    if not dvc.is_initialized:
        print("\nInitializing DVC...")
        success = dvc.init()
        if success:
            print("✓ DVC initialized successfully")
        else:
            print("✗ Failed to initialize DVC")
            return
    else:
        print("\n✓ DVC already initialized")
    
    temp_data = np.random.rand(50, 5)
    temp_file = Path("dvc_dataset.npy")
    np.save(temp_file, temp_data)
    
    print(f"\nAdding {temp_file} to DVC...")
    success = dvc.add(temp_file)
    
    if success:
        print(f"✓ Added {temp_file} to DVC")
        
        tracked_files = dvc.get_tracked_files()
        print(f"\nTracked files: {len(tracked_files)}")
        for f in tracked_files:
            print(f"  - {f}")
    else:
        print(f"✗ Failed to add {temp_file} to DVC")
    
    if temp_file.exists():
        temp_file.unlink()
    
    dvc_file = temp_file.with_suffix(temp_file.suffix + ".dvc")
    if dvc_file.exists():
        dvc_file.unlink()
    
    print("\n✓ DVC integration example completed!")


def example_preprocessing_pipeline():
    print("\n" + "=" * 60)
    print("Preprocessing Pipeline Example")
    print("=" * 60)
    
    base_dir = Path(".neural_data_example")
    tracker = PreprocessingTracker(base_dir=base_dir)
    
    pipeline = tracker.create_pipeline(
        name="mnist_pipeline",
        description="Standard MNIST preprocessing"
    )
    
    pipeline.add_step(
        name="normalize",
        params={"min": 0, "max": 255},
        description="Normalize pixel values to [0, 1]"
    )
    
    pipeline.add_step(
        name="reshape",
        params={"shape": [28, 28, 1]},
        description="Reshape to image format"
    )
    
    pipeline.add_step(
        name="augment",
        params={"rotation": 15, "flip": True},
        description="Data augmentation"
    )
    
    tracker.update_pipeline(pipeline)
    
    print(f"\nCreated pipeline: {pipeline.name}")
    print(f"Steps: {len(pipeline.steps)}")
    for step in pipeline.steps:
        print(f"  - {step.name}: {step.description}")
    
    pipeline2 = tracker.create_pipeline(
        name="cifar_pipeline",
        description="CIFAR-10 preprocessing"
    )
    
    pipeline2.add_step(
        name="normalize",
        params={"min": 0, "max": 255, "mean": [0.5, 0.5, 0.5]},
        description="Normalize with mean subtraction"
    )
    
    tracker.update_pipeline(pipeline2)
    
    comparison = tracker.compare_pipelines("mnist_pipeline", "cifar_pipeline")
    print(f"\nPipeline comparison:")
    print(f"  Same steps: {comparison['same_steps']}")
    print(f"  Differences: {len(comparison['differences'])}")
    
    print("\n✓ Preprocessing pipeline example completed!")


def example_feature_store():
    print("\n" + "=" * 60)
    print("Feature Store Example")
    print("=" * 60)
    
    base_dir = Path(".neural_data_example")
    store = FeatureStore(base_dir=base_dir)
    
    image_group = store.create_feature_group(
        name="image_features",
        description="Features extracted from images"
    )
    
    store.add_feature(
        "image_features",
        "pixel_mean",
        "float32",
        "Mean pixel value",
        {"range": [0, 255]}
    )
    
    store.add_feature(
        "image_features",
        "pixel_std",
        "float32",
        "Standard deviation of pixels"
    )
    
    store.add_feature(
        "image_features",
        "edge_density",
        "float32",
        "Density of edges in image"
    )
    
    text_group = store.create_feature_group(
        name="text_features",
        description="Features from text data"
    )
    
    store.add_feature(
        "text_features",
        "word_count",
        "int32",
        "Number of words"
    )
    
    store.add_feature(
        "text_features",
        "sentiment_score",
        "float32",
        "Sentiment analysis score",
        {"range": [-1, 1]}
    )
    
    print(f"\nCreated feature groups:")
    for group_name in store.list_feature_groups():
        group = store.get_feature_group(group_name)
        print(f"  - {group.name}: {len(group.features)} features")
    
    stats = store.get_feature_statistics()
    print(f"\nFeature Store Statistics:")
    print(f"  Total groups: {stats['total_groups']}")
    print(f"  Total features: {stats['total_features']}")
    print(f"  Data types: {stats['dtype_distribution']}")
    
    results = store.search_features("pixel")
    print(f"\nSearch results for 'pixel': {len(results)} features")
    for result in results:
        print(f"  - {result['group']}.{result['feature']} ({result['dtype']})")
    
    print("\n✓ Feature store example completed!")


def example_quality_validation():
    print("\n" + "=" * 60)
    print("Data Quality Validation Example")
    print("=" * 60)
    
    base_dir = Path(".neural_data_example")
    validator = DataQualityValidator(base_dir=base_dir)
    
    test_data = np.random.rand(100, 10)
    
    print("\nValidating clean data...")
    results = validator.validate(test_data)
    
    passed = sum(1 for r in results if r.passed)
    print(f"Results: {passed}/{len(results)} rules passed")
    
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.rule_name}")
    
    test_data_with_nan = test_data.copy()
    test_data_with_nan[0, 0] = np.nan
    
    print("\nValidating data with missing values...")
    results = validator.validate(test_data_with_nan)
    
    passed = sum(1 for r in results if r.passed)
    print(f"Results: {passed}/{len(results)} rules passed")
    
    for result in results:
        if not result.passed:
            status = "✗"
            print(f"  {status} {result.rule_name}: {result.message}")
    
    validator.create_custom_rule(
        name="min_100_rows",
        rule_type="size",
        condition="min_rows",
        threshold=100,
        description="Dataset must have at least 100 rows"
    )
    
    print("\nCustom rule created: min_100_rows")
    
    print("\n✓ Quality validation example completed!")


def example_lineage_tracking():
    print("\n" + "=" * 60)
    print("Lineage Tracking Example")
    print("=" * 60)
    
    base_dir = Path(".neural_data_example")
    
    graph_name = create_data_to_model_lineage(
        graph_name="mnist_experiment",
        dataset_version="v1.0",
        preprocessing_pipeline="mnist_pipeline",
        model_name="cnn_classifier",
        prediction_name="test_predictions",
        base_dir=base_dir,
    )
    
    print(f"\nCreated lineage graph: {graph_name}")
    
    tracker = LineageTracker(base_dir=base_dir)
    graph = tracker.get_graph(graph_name)
    
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    
    print("\nNodes:")
    for node in graph.nodes.values():
        print(f"  - [{node.node_type}] {node.name}")
    
    print("\nEdges:")
    for edge in graph.edges:
        source = graph.get_node(edge.source_id)
        target = graph.get_node(edge.target_id)
        print(f"  - {source.name} --[{edge.edge_type}]--> {target.name}")
    
    data_node_id = "data_v1.0"
    lineage = tracker.get_full_lineage(graph_name, data_node_id)
    
    print(f"\nLineage from {data_node_id}:")
    print(f"  Upstream: {len(lineage['upstream'])} nodes")
    print(f"  Downstream: {len(lineage['downstream'])} nodes")
    
    try:
        viz_path = tracker.visualize_lineage(graph_name)
        print(f"\nLineage visualization saved to: {viz_path}")
    except Exception as e:
        print(f"\nVisualization requires graphviz: {str(e)}")
        print("Install with: pip install graphviz")
    
    print("\n✓ Lineage tracking example completed!")


def run_all_examples():
    print("\n" + "=" * 60)
    print("Neural Data Versioning & Lineage Examples")
    print("=" * 60)
    
    example_dataset_versioning()
    example_dvc_integration()
    example_preprocessing_pipeline()
    example_feature_store()
    example_quality_validation()
    example_lineage_tracking()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    import shutil
    example_dir = Path(".neural_data_example")
    if example_dir.exists():
        shutil.rmtree(example_dir)
        print("\nCleaned up example data directory")


if __name__ == "__main__":
    run_all_examples()
