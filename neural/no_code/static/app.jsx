const { useState, useEffect, useCallback, useRef } = React;
const { ReactFlow, Background, Controls, MiniMap, useNodesState, useEdgesState, addEdge, MarkerType } = ReactFlowRenderer;

const API_BASE = '';

function App() {
    const [activeTab, setActiveTab] = useState('layers');
    const [layers, setLayers] = useState({});
    const [templates, setTemplates] = useState({});
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedNode, setSelectedNode] = useState(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [validation, setValidation] = useState({ valid: true, errors: [], warnings: [], shapes: [] });
    const [showExportModal, setShowExportModal] = useState(false);
    const [showTemplateModal, setShowTemplateModal] = useState(false);
    const [showTutorial, setShowTutorial] = useState(false);
    const [tutorialStep, setTutorialStep] = useState(0);
    const [tutorialSteps, setTutorialSteps] = useState([]);
    const [generatedCode, setGeneratedCode] = useState({ dsl: '', tensorflow: '', pytorch: '' });
    const [codeTab, setCodeTab] = useState('dsl');
    const [expandedCategories, setExpandedCategories] = useState({});
    const nodeIdCounter = useRef(0);

    useEffect(() => {
        fetch(`${API_BASE}/api/layers`)
            .then(res => res.json())
            .then(data => {
                setLayers(data);
                const expanded = {};
                Object.keys(data).forEach(cat => expanded[cat] = true);
                setExpandedCategories(expanded);
            });

        fetch(`${API_BASE}/api/templates`)
            .then(res => res.json())
            .then(data => setTemplates(data));

        fetch(`${API_BASE}/api/tutorial`)
            .then(res => res.json())
            .then(data => setTutorialSteps(data));
    }, []);

    useEffect(() => {
        validateModel();
    }, [nodes]);

    const validateModel = async () => {
        if (nodes.length === 0) {
            setValidation({ valid: true, errors: [], warnings: [{ type: 'warning', message: 'Model is empty' }], shapes: [] });
            return;
        }

        const modelLayers = nodes
            .sort((a, b) => a.position.y - b.position.y)
            .map(node => ({
                id: node.id,
                type: node.data.layerType,
                params: node.data.params || {}
            }));

        const response = await fetch(`${API_BASE}/api/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input_shape: [null, 28, 28, 1],
                layers: modelLayers
            })
        });

        const result = await response.json();
        setValidation(result);

        const updatedNodes = nodes.map(node => {
            const hasError = result.errors.some(err => err.layer_id === node.id);
            return {
                ...node,
                className: hasError ? 'error' : ''
            };
        });
        setNodes(updatedNodes);
    };

    const onConnect = useCallback((params) => {
        setEdges((eds) => addEdge({ ...params, type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed } }, eds));
    }, [setEdges]);

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback((event) => {
        event.preventDefault();

        const layerData = JSON.parse(event.dataTransfer.getData('application/reactflow'));
        const position = {
            x: event.clientX - event.target.getBoundingClientRect().left,
            y: event.clientY - event.target.getBoundingClientRect().top,
        };

        const newNode = {
            id: `node-${nodeIdCounter.current++}`,
            type: 'custom',
            position,
            data: {
                layerType: layerData.name,
                params: { ...layerData.params },
                onDelete: (id) => setNodes(nds => nds.filter(n => n.id !== id))
            },
        };

        setNodes((nds) => nds.concat(newNode));
    }, [setNodes]);

    const onNodeClick = useCallback((event, node) => {
        setSelectedNode(node);
    }, []);

    const updateNodeParams = (nodeId, params) => {
        setNodes(nds => nds.map(node => {
            if (node.id === nodeId) {
                return {
                    ...node,
                    data: { ...node.data, params }
                };
            }
            return node;
        }));
    };

    const loadTemplate = (templateKey) => {
        const template = templates[templateKey];
        if (!template) return;

        const newNodes = template.layers.map((layer, index) => ({
            id: `node-${nodeIdCounter.current++}`,
            type: 'custom',
            position: { x: 250, y: index * 120 + 50 },
            data: {
                layerType: layer.type,
                params: { ...layer.params },
                onDelete: (id) => setNodes(nds => nds.filter(n => n.id !== id))
            }
        }));

        const newEdges = newNodes.slice(0, -1).map((node, index) => ({
            id: `edge-${index}`,
            source: node.id,
            target: newNodes[index + 1].id,
            type: 'smoothstep',
            markerEnd: { type: MarkerType.ArrowClosed }
        }));

        setNodes(newNodes);
        setEdges(newEdges);
        setShowTemplateModal(false);
    };

    const generateCode = async () => {
        const modelLayers = nodes
            .sort((a, b) => a.position.y - b.position.y)
            .map(node => ({
                type: node.data.layerType,
                params: node.data.params || {}
            }));

        const response = await fetch(`${API_BASE}/api/generate-code`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input_shape: [null, 28, 28, 1],
                layers: modelLayers,
                optimizer: { type: 'Adam', params: { learning_rate: 0.001 } },
                loss: 'categorical_crossentropy'
            })
        });

        const result = await response.json();
        setGeneratedCode(result);
        setShowExportModal(true);
    };

    const filteredLayers = () => {
        if (!searchTerm) return layers;

        const filtered = {};
        Object.entries(layers).forEach(([category, layerList]) => {
            const matching = layerList.filter(layer =>
                layer.name.toLowerCase().includes(searchTerm.toLowerCase())
            );
            if (matching.length > 0) {
                filtered[category] = matching;
            }
        });
        return filtered;
    };

    const toggleCategory = (category) => {
        setExpandedCategories(prev => ({
            ...prev,
            [category]: !prev[category]
        }));
    };

    return (
        <div className="app-container">
            <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
                <div className="app-header">
                    <div className="app-title">
                        ⚡ Neural DSL - No-Code Designer
                    </div>
                    <div className="header-buttons">
                        <button className="btn btn-secondary" onClick={() => setShowTutorial(!showTutorial)}>
                            📖 Tutorial
                        </button>
                        <button className="btn btn-primary" onClick={() => setShowTemplateModal(true)}>
                            📋 Templates
                        </button>
                        <button className="btn btn-success" onClick={generateCode}>
                            🚀 Export Code
                        </button>
                    </div>
                </div>

                <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                    <div className="sidebar">
                        <div className="sidebar-tabs">
                            <button
                                className={`sidebar-tab ${activeTab === 'layers' ? 'active' : ''}`}
                                onClick={() => setActiveTab('layers')}
                            >
                                Layers
                            </button>
                            <button
                                className={`sidebar-tab ${activeTab === 'properties' ? 'active' : ''}`}
                                onClick={() => setActiveTab('properties')}
                            >
                                Properties
                            </button>
                        </div>

                        <div className="sidebar-content">
                            {activeTab === 'layers' && (
                                <LayerPalette
                                    layers={filteredLayers()}
                                    searchTerm={searchTerm}
                                    onSearchChange={setSearchTerm}
                                    expandedCategories={expandedCategories}
                                    onToggleCategory={toggleCategory}
                                />
                            )}
                            {activeTab === 'properties' && (
                                <PropertiesPanel
                                    selectedNode={selectedNode}
                                    onUpdateParams={updateNodeParams}
                                />
                            )}
                        </div>
                    </div>

                    <div className="main-content">
                        <div className="flow-canvas" onDrop={onDrop} onDragOver={onDragOver}>
                            <ReactFlow
                                nodes={nodes}
                                edges={edges}
                                onNodesChange={onNodesChange}
                                onEdgesChange={onEdgesChange}
                                onConnect={onConnect}
                                onNodeClick={onNodeClick}
                                nodeTypes={{ custom: CustomNode }}
                                fitView
                            >
                                <Background />
                                <Controls />
                                <MiniMap />
                            </ReactFlow>
                            {nodes.length === 0 && (
                                <div className="empty-state">
                                    <div className="empty-state-icon">🎨</div>
                                    <div className="empty-state-title">Start Building</div>
                                    <div className="empty-state-text">
                                        Drag layers from the palette or load a template
                                    </div>
                                </div>
                            )}
                        </div>

                        <ValidationPanel validation={validation} />
                    </div>
                </div>
            </div>

            {showExportModal && (
                <ExportModal
                    code={generatedCode}
                    codeTab={codeTab}
                    onCodeTabChange={setCodeTab}
                    onClose={() => setShowExportModal(false)}
                />
            )}

            {showTemplateModal && (
                <TemplateModal
                    templates={templates}
                    onLoad={loadTemplate}
                    onClose={() => setShowTemplateModal(false)}
                />
            )}

            {showTutorial && tutorialSteps.length > 0 && (
                <Tutorial
                    steps={tutorialSteps}
                    currentStep={tutorialStep}
                    onNext={() => setTutorialStep(s => Math.min(s + 1, tutorialSteps.length - 1))}
                    onPrev={() => setTutorialStep(s => Math.max(s - 1, 0))}
                    onClose={() => setShowTutorial(false)}
                />
            )}
        </div>
    );
}

function CustomNode({ data }) {
    const shapeInfo = data.shape ? `Shape: ${JSON.stringify(data.shape)}` : '';
    const paramsStr = Object.entries(data.params || {})
        .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
        .join(', ');

    return (
        <div>
            <div className="node-header">
                <span>{data.layerType}</span>
                <button className="node-delete" onClick={() => data.onDelete(data.id)}>×</button>
            </div>
            {paramsStr && <div className="node-params">{paramsStr}</div>}
            {shapeInfo && <div className="node-shape">{shapeInfo}</div>}
        </div>
    );
}

function LayerPalette({ layers, searchTerm, onSearchChange, expandedCategories, onToggleCategory }) {
    const onDragStart = (event, layer) => {
        event.dataTransfer.setData('application/reactflow', JSON.stringify(layer));
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <div className="layer-palette">
            <input
                type="text"
                className="search-box"
                placeholder="🔍 Search layers..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
            />

            {Object.entries(layers).map(([category, layerList]) => (
                <div key={category} className="layer-category">
                    <div className="category-header" onClick={() => onToggleCategory(category)}>
                        <span>{category}</span>
                        <span>{expandedCategories[category] ? '▼' : '▶'}</span>
                    </div>
                    {expandedCategories[category] && (
                        <div className="layer-list">
                            {layerList.map(layer => (
                                <div
                                    key={layer.name}
                                    className="layer-item"
                                    draggable
                                    onDragStart={(e) => onDragStart(e, layer)}
                                >
                                    <div className="layer-name">{layer.name}</div>
                                    <div className="layer-description">
                                        {Object.keys(layer.params).slice(0, 2).join(', ')}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}

function PropertiesPanel({ selectedNode, onUpdateParams }) {
    const [params, setParams] = useState({});

    useEffect(() => {
        if (selectedNode) {
            setParams(selectedNode.data.params || {});
        }
    }, [selectedNode]);

    const handleParamChange = (key, value) => {
        const newParams = { ...params, [key]: value };
        setParams(newParams);
        if (selectedNode) {
            onUpdateParams(selectedNode.id, newParams);
        }
    };

    if (!selectedNode) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">⚙️</div>
                <div className="empty-state-title">No Layer Selected</div>
                <div className="empty-state-text">Click a layer to edit its properties</div>
            </div>
        );
    }

    return (
        <div>
            <h3 className="mb-2">{selectedNode.data.layerType}</h3>
            {Object.entries(params).map(([key, value]) => (
                <div key={key} className="property-group">
                    <label className="property-label">{key}</label>
                    <input
                        type="text"
                        className="property-input"
                        value={JSON.stringify(value)}
                        onChange={(e) => {
                            try {
                                const parsed = JSON.parse(e.target.value);
                                handleParamChange(key, parsed);
                            } catch {
                                handleParamChange(key, e.target.value);
                            }
                        }}
                    />
                </div>
            ))}
        </div>
    );
}

function ValidationPanel({ validation }) {
    const hasErrors = validation.errors.length > 0;
    const hasWarnings = validation.warnings.length > 0;

    return (
        <div className="validation-panel">
            <div className="validation-header">
                <div className="validation-title">Validation</div>
                <div className="validation-status">
                    <span className={`status-icon ${hasErrors ? 'error' : hasWarnings ? 'warning' : 'success'}`}></span>
                    <span>{hasErrors ? 'Errors' : hasWarnings ? 'Warnings' : 'Valid'}</span>
                </div>
            </div>

            <div className="validation-messages">
                {validation.errors.map((err, i) => (
                    <div key={i} className="validation-message error">
                        <span>❌</span>
                        <span>{err.message}</span>
                    </div>
                ))}
                {validation.warnings.map((warn, i) => (
                    <div key={i} className="validation-message warning">
                        <span>⚠️</span>
                        <span>{warn.message}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ExportModal({ code, codeTab, onCodeTabChange, onClose }) {
    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
    };

    const codeMap = {
        dsl: { label: 'Neural DSL', code: code.dsl },
        tensorflow: { label: 'TensorFlow', code: code.tensorflow },
        pytorch: { label: 'PyTorch', code: code.pytorch }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div className="modal-title">Export Code</div>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>
                <div className="modal-body">
                    <div className="code-tabs">
                        {Object.entries(codeMap).map(([key, { label }]) => (
                            <button
                                key={key}
                                className={`code-tab ${codeTab === key ? 'active' : ''}`}
                                onClick={() => onCodeTabChange(key)}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                    <div className="code-block">{codeMap[codeTab].code}</div>
                </div>
                <div className="modal-footer">
                    <button className="btn btn-secondary" onClick={() => copyToClipboard(codeMap[codeTab].code)}>
                        📋 Copy
                    </button>
                    <button className="btn btn-primary" onClick={onClose}>Close</button>
                </div>
            </div>
        </div>
    );
}

function TemplateModal({ templates, onLoad, onClose }) {
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div className="modal-title">Model Templates</div>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>
                <div className="modal-body">
                    <div className="templates-grid">
                        {Object.entries(templates).map(([key, template]) => (
                            <div
                                key={key}
                                className="template-card"
                                onClick={() => onLoad(key)}
                            >
                                <div className="template-name">{template.name}</div>
                                <div className="template-description">{template.description}</div>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="modal-footer">
                    <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
                </div>
            </div>
        </div>
    );
}

function Tutorial({ steps, currentStep, onNext, onPrev, onClose }) {
    const step = steps[currentStep];

    return (
        <>
            <div className="tutorial-overlay"></div>
            <div className="tutorial-popup" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                <div className="tutorial-content">
                    <div className="tutorial-title">{step.title}</div>
                    <div className="tutorial-text">{step.content}</div>
                </div>
                <div className="tutorial-buttons">
                    <div className="tutorial-progress">
                        {currentStep + 1} / {steps.length}
                    </div>
                    <div className="tutorial-actions">
                        {currentStep > 0 && (
                            <button className="btn btn-secondary" onClick={onPrev}>Previous</button>
                        )}
                        {currentStep < steps.length - 1 ? (
                            <button className="btn btn-primary" onClick={onNext}>Next</button>
                        ) : (
                            <button className="btn btn-success" onClick={onClose}>Finish</button>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
