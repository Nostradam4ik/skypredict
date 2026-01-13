import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, Plane, Clock, AlertTriangle, BarChart2 } from 'lucide-react';
import { HistoryItem } from './PredictionHistory';

interface StatsDashboardProps {
  history: HistoryItem[];
}

const StatsDashboard: React.FC<StatsDashboardProps> = ({ history }) => {
  if (history.length === 0) {
    return null;
  }

  // Calculate statistics
  const totalPredictions = history.length;
  const delayedCount = history.filter(h => h.prediction.is_delayed).length;
  const onTimeCount = totalPredictions - delayedCount;
  const avgDelayProb = history.reduce((acc, h) => acc + h.prediction.delay_probability, 0) / totalPredictions;

  // Risk level distribution
  const riskDistribution = {
    LOW: history.filter(h => h.prediction.risk_level === 'LOW').length,
    MEDIUM: history.filter(h => h.prediction.risk_level === 'MEDIUM').length,
    HIGH: history.filter(h => h.prediction.risk_level === 'HIGH').length,
    SEVERE: history.filter(h => h.prediction.risk_level === 'SEVERE').length
  };

  const riskData = [
    { name: 'Low', value: riskDistribution.LOW, color: '#22c55e' },
    { name: 'Medium', value: riskDistribution.MEDIUM, color: '#f59e0b' },
    { name: 'High', value: riskDistribution.HIGH, color: '#ef4444' },
    { name: 'Severe', value: riskDistribution.SEVERE, color: '#7c2d12' }
  ].filter(d => d.value > 0);

  // Hourly trend data
  const hourlyData: Record<number, { count: number; totalProb: number }> = {};
  history.forEach(h => {
    const hour = parseInt(h.prediction.scheduled_departure.split(':')[0]);
    if (!hourlyData[hour]) {
      hourlyData[hour] = { count: 0, totalProb: 0 };
    }
    hourlyData[hour].count++;
    hourlyData[hour].totalProb += h.prediction.delay_probability;
  });

  const hourlyTrend = Object.entries(hourlyData)
    .map(([hour, data]) => ({
      hour: `${hour}h`,
      avgRisk: Math.round((data.totalProb / data.count) * 100)
    }))
    .sort((a, b) => parseInt(a.hour) - parseInt(b.hour));

  // Top airports
  const airportStats: Record<string, { count: number; totalProb: number }> = {};
  history.forEach(h => {
    [h.prediction.origin_airport, h.prediction.dest_airport].forEach(airport => {
      if (!airportStats[airport]) {
        airportStats[airport] = { count: 0, totalProb: 0 };
      }
      airportStats[airport].count++;
      airportStats[airport].totalProb += h.prediction.delay_probability;
    });
  });

  const topAirports = Object.entries(airportStats)
    .map(([code, data]) => ({
      code,
      avgRisk: Math.round((data.totalProb / data.count) * 100),
      flights: data.count
    }))
    .sort((a, b) => b.flights - a.flights)
    .slice(0, 5);

  return (
    <div className="stats-dashboard">
      <div className="stats-header">
        <h3><BarChart2 size={20} /> Analytics Dashboard</h3>
      </div>

      <div className="stats-summary">
        <div className="summary-card">
          <Plane size={24} />
          <div className="summary-info">
            <span className="summary-value">{totalPredictions}</span>
            <span className="summary-label">Predictions</span>
          </div>
        </div>
        <div className="summary-card">
          <TrendingUp size={24} />
          <div className="summary-info">
            <span className="summary-value">{Math.round(avgDelayProb * 100)}%</span>
            <span className="summary-label">Avg Risk</span>
          </div>
        </div>
        <div className="summary-card success">
          <Clock size={24} />
          <div className="summary-info">
            <span className="summary-value">{onTimeCount}</span>
            <span className="summary-label">On Time</span>
          </div>
        </div>
        <div className="summary-card danger">
          <AlertTriangle size={24} />
          <div className="summary-info">
            <span className="summary-value">{delayedCount}</span>
            <span className="summary-label">Delayed</span>
          </div>
        </div>
      </div>

      <div className="stats-charts">
        <div className="chart-card">
          <h4>Risk Distribution</h4>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                innerRadius={35}
                outerRadius={55}
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => [`${value} flights`, '']}
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.5rem'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="chart-legend">
            {riskData.map((entry) => (
              <span key={entry.name} className="legend-item">
                <span className="legend-dot" style={{ background: entry.color }} />
                {entry.name}
              </span>
            ))}
          </div>
        </div>

        {hourlyTrend.length > 1 && (
          <div className="chart-card">
            <h4>Hourly Trend</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={hourlyTrend}>
                <XAxis
                  dataKey="hour"
                  tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
                  axisLine={{ stroke: 'var(--border)' }}
                />
                <YAxis
                  tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
                  axisLine={{ stroke: 'var(--border)' }}
                  domain={[0, 100]}
                />
                <Tooltip
                  formatter={(value) => [`${value}%`, 'Avg Risk']}
                  contentStyle={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    borderRadius: '0.5rem'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="avgRisk"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {topAirports.length > 0 && (
        <div className="chart-card full-width">
          <h4>Top Airports</h4>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={topAirports} layout="vertical">
              <XAxis type="number" domain={[0, 100]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
              <YAxis
                type="category"
                dataKey="code"
                tick={{ fill: 'var(--text-primary)', fontSize: 11, fontWeight: 600 }}
                width={50}
              />
              <Tooltip
                formatter={(value) => [`${value}%`, 'Avg Risk']}
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.5rem'
                }}
              />
              <Bar
                dataKey="avgRisk"
                fill="#3b82f6"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default StatsDashboard;
