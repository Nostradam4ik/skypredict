import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip
} from 'recharts';

interface ProbabilityChartProps {
  probability: number;
  riskLevel: string;
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probability, riskLevel }) => {
  const delayProb = Math.round(probability * 100);
  const onTimeProb = 100 - delayProb;

  const data = [
    { name: 'On Time', value: onTimeProb, color: '#22c55e' },
    { name: 'Delayed', value: delayProb, color: getRiskColor(riskLevel) }
  ];

  function getRiskColor(risk: string): string {
    switch (risk) {
      case 'LOW': return '#f59e0b';
      case 'MEDIUM': return '#f97316';
      case 'HIGH': return '#ef4444';
      case 'SEVERE': return '#7c2d12';
      default: return '#f59e0b';
    }
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p>{`${payload[0].name}: ${payload[0].value}%`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="probability-chart">
      <h4>Delay Probability</h4>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={2}
            dataKey="value"
            animationBegin={0}
            animationDuration={800}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value) => <span style={{ color: '#94a3b8' }}>{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="chart-center-label">
        <span className="center-value">{delayProb}%</span>
        <span className="center-text">delay risk</span>
      </div>
    </div>
  );
};

export default ProbabilityChart;
