const CORE_QUBITS = [
  { id: 'Q0', role: 'Logic', color: 0x9b5cff },
  { id: 'Q1', role: 'Emotion', color: 0x7c5cff },
  { id: 'Q2', role: 'Intuition', color: 0x4f72ff },
  { id: 'Q3', role: 'Memory', color: 0x3daeff },
  { id: 'Q4', role: 'Sovereignty', color: 0x56d7ff },
  { id: 'Q5', role: 'Attention', color: 0x78d88b },
  { id: 'Q6', role: 'Reflection', color: 0xd8d65f },
  { id: 'Q7', role: 'Language', color: 0xffcf4f },
  { id: 'Q8', role: 'Planning', color: 0xffa24a },
  { id: 'Q9', role: 'Novelty', color: 0xffd76a },
  { id: 'Q10', role: 'Stability', color: 0xff914d },
  { id: 'Q11', role: 'Meta-Reasoning', color: 0xd83cff },
];
const QUANTUM_ARCHITECTURE = {
  summary: {
    label: '156Q ARK',
    detail: '12 core + 144 reservoir',
    coreQubits: 12,
    reservoirQubits: 144,
    topology: 'dodecahedral core / akashic reservoir / readout membrane',
  },
  modules: [
    {
      key: 'encoding',
      label: 'STATE ENCODING',
      detail: 'basis -> phase',
      color: 0x3daeff,
      anchor: [-3.75, -2.42, 1.18],
      labelAnchor: [-2.92, -1.35, 1.42],
      targets: [0, 2, 3, 7, 11],
    },
    {
      key: 'dynamics',
      label: 'OPEN SYSTEM',
      detail: 'drive + decay',
      color: 0x78d88b,
      anchor: [0, 3.84, -0.55],
      labelAnchor: [0, 2.54, 0.28],
      targets: [4, 5, 6, 10],
    },
    {
      key: 'entropy',
      label: 'ENTROPY ROUTING',
      detail: 'heat -> sinks',
      color: 0xffcf4f,
      anchor: [3.76, -2.28, 1.05],
      labelAnchor: [2.92, -1.34, 1.42],
      targets: [4, 5, 6, 10],
    },
    {
      key: 'readout',
      label: 'MEASUREMENT',
      detail: 'projection readout',
      color: 0xeef1f6,
      anchor: [0, -3.86, 2.2],
      labelAnchor: [0, -2.44, 2.26],
      targets: [0, 7, 8, 11],
    },
  ],
  topologicalLayers: [
    { key: 'core', label: 'L0 CORE', radius: 2.1, color: 0xb679ff, tilt: [0.42, 0.0, 0.0] },
    { key: 'coupling', label: 'L1 COUPLING', radius: 2.78, color: 0xffe39a, tilt: [1.28, 0.18, 0.0] },
    { key: 'reservoir', label: 'L2 RESERVOIR', radius: 3.54, color: 0x31cfff, tilt: [0.0, 0.52, 0.0] },
    { key: 'readout', label: 'L3 READOUT', radius: 4.08, color: 0xeef1f6, tilt: [1.56, 0.0, 0.28] },
  ],
};

export { CORE_QUBITS, QUANTUM_ARCHITECTURE };
