import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { auth, db, isConfigured } from './firebase';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged 
} from 'firebase/auth';
import { 
  collection, 
  onSnapshot, 
  query 
} from 'firebase/firestore';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function App() {
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [authError, setAuthError] = useState('');

  const [showSplash, setShowSplash] = useState(true);
  const [splashFade, setSplashFade] = useState(false);
  const [splashMessage, setSplashMessage] = useState('Establishing connection to AI metrics database...');
  const [activeTab, setActiveTab] = useState('overview');
  
  const [records, setRecords] = useState([]);
  const [allDepartments, setAllDepartments] = useState([]);
  const [selectedDepts, setSelectedDepts] = useState([]);
  const [dateRange, setDateRange] = useState({ min: '', max: '', start: '', end: '' });
  
  const [forecastDept, setForecastDept] = useState('');
  const [forecastKpi, setForecastKpi] = useState('attrition_rate');
  const [forecastList, setForecastList] = useState([]);
  const [forecastLoading, setForecastLoading] = useState(false);

  const [summaryInput, setSummaryInput] = useState('');
  const [summaryResult, setSummaryResult] = useState('');
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [sentimentInput, setSentimentInput] = useState('');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState({ text: '', type: '' });
  const [uploading, setUploading] = useState(false);

  const [loading, setLoading] = useState(true);

  const riskLevelCheck = (r) => {
    const attr = parseFloat(r.attrition_rate) || 0;
    const eng = parseFloat(r.engagement_score) || 0;
    const delay = parseFloat(r.project_delay_index) || 0;
    if (attr > 0.15 || eng < 0.7 || delay > 0.2) {
      return "High";
    } else if (attr > 0.10 || eng < 0.8 || delay > 0.1) {
      return "Medium";
    } else {
      return "Low";
    }
  };

  const processFirestoreDocs = (docs) => {
    const rawList = docs.map(d => ({
      ...d,
      attrition_rate: parseFloat(d.attrition_rate) || 0,
      engagement_score: parseFloat(d.engagement_score) || 0,
      project_delay_index: parseFloat(d.project_delay_index) || 0
    }));
    
    rawList.sort((a, b) => {
      if (a.department !== b.department) return a.department.localeCompare(b.department);
      return a.date.localeCompare(b.date);
    });
    
    const deptGroups = {};
    const processed = rawList.map(r => {
      const dept = r.department;
      if (!deptGroups[dept]) {
        deptGroups[dept] = { lastAttrition: null, lastEngagement: null };
      }
      
      const attritionChange = deptGroups[dept].lastAttrition !== null ? (r.attrition_rate - deptGroups[dept].lastAttrition) : 0;
      const engagementChange = deptGroups[dept].lastEngagement !== null ? (r.engagement_score - deptGroups[dept].lastEngagement) : 0;
      
      deptGroups[dept].lastAttrition = r.attrition_rate;
      deptGroups[dept].lastEngagement = r.engagement_score;
      
      const risk = riskLevelCheck(r);
      
      return {
        ...r,
        predicted_risk: risk,
        sentiment_label: r.sentiment_label || "NEUTRAL",
        sentiment_score: r.sentiment_score || 0.5,
        attrition_change: attritionChange,
        engagement_change: engagementChange,
        meeting_summary: r.meeting_summary || r.meeting_transcript
      };
    });

    setRecords(processed);
    
    const depts = [...new Set(processed.map(p => p.department))];
    setAllDepartments(depts);
    
    if (depts.length > 0 && !forecastDept) {
      setForecastDept(depts[0]);
    }
    
    if (processed.length > 0 && !dateRange.min) {
      const dates = processed.map(p => p.date);
      const minD = dates.reduce((a, b) => a < b ? a : b);
      const maxD = dates.reduce((a, b) => a > b ? a : b);
      setDateRange({
        min: minD,
        max: maxD,
        start: minD,
        end: maxD
      });
    }
  };

  const fetchMetrics = async (deptsFilter = selectedDepts, start = dateRange.start, end = dateRange.end) => {
    try {
      setLoading(true);
      const deptQuery = deptsFilter.length > 0 ? `?departments=${deptsFilter.join(',')}` : '';
      const startQuery = start ? `${deptQuery ? '&' : '?'}start_date=${start}` : '';
      const endQuery = end ? `${(deptQuery || startQuery) ? '&' : '?'}end_date=${end}` : '';
      
      const response = await fetch(`/api/metrics${deptQuery}${startQuery}${endQuery}`);
      const data = await response.json();
      
      if (response.ok) {
        setRecords(data.records);
        setAllDepartments(data.all_departments);
        
        if (!dateRange.min) {
          setDateRange({
            min: data.min_date,
            max: data.max_date,
            start: data.min_date,
            end: data.max_date
          });
        }
        
        if (selectedDepts.length === 0 && deptsFilter.length === 0) {
          setSelectedDepts(data.all_departments);
          if (data.all_departments.length > 0) {
            setForecastDept(data.all_departments[0]);
          }
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isConfigured) {
      const unsubscribeAuth = onAuthStateChanged(auth, (usr) => {
        setUser(usr);
        if (usr) {
          const q = query(collection(db, "metrics"));
          const unsubscribeFirestore = onSnapshot(q, (snapshot) => {
            const docsList = [];
            snapshot.forEach((doc) => {
              docsList.push(doc.data());
            });
            processFirestoreDocs(docsList);
            setLoading(false);
          });
          
          setTimeout(() => {
            setSplashFade(true);
            setTimeout(() => {
              setShowSplash(false);
            }, 600);
          }, 1500);

          return () => unsubscribeFirestore();
        } else {
          setLoading(false);
          setShowSplash(false);
        }
      });
      return () => unsubscribeAuth();
    } else {
      setUser({ email: 'sandbox@smartcorp.ai', displayName: 'Sandbox User' });
      const runStartup = async () => {
        await fetchMetrics([], '', '');
        setTimeout(() => {
          setSplashFade(true);
          setTimeout(() => {
            setShowSplash(false);
          }, 600);
        }, 1500);
      };
      runStartup();
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!email || !password) return;
    try {
      setAuthError('');
      setLoading(true);
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!email || !password) return;
    try {
      setAuthError('');
      setLoading(true);
      await createUserWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    if (isConfigured) {
      await signOut(auth);
    } else {
      setUser(null);
    }
  };

  const handleDeptToggle = (dept) => {
    let updated;
    if (selectedDepts.includes(dept)) {
      updated = selectedDepts.filter(d => d !== dept);
    } else {
      updated = [...selectedDepts, dept];
    }
    setSelectedDepts(updated);
    if (!isConfigured) {
      fetchMetrics(updated, dateRange.start, dateRange.end);
    }
  };

  const handleDateChange = (type, val) => {
    const updatedRange = { ...dateRange, [type]: val };
    setDateRange(updatedRange);
    if (!isConfigured) {
      fetchMetrics(selectedDepts, updatedRange.start, updatedRange.end);
    }
  };

  const resetFilters = () => {
    const resetRange = { ...dateRange, start: dateRange.min, end: dateRange.max };
    setDateRange(resetRange);
    setSelectedDepts(allDepartments);
    if (!isConfigured) {
      fetchMetrics(allDepartments, resetRange.start, resetRange.end);
    }
  };

  const loadForecast = async () => {
    if (!forecastDept || !forecastKpi) return;
    try {
      setForecastLoading(true);
      const response = await fetch('/api/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ department: forecastDept, kpi: forecastKpi })
      });
      const data = await response.json();
      if (response.ok) {
        setForecastList(data.forecast);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setForecastLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'forecast') {
      loadForecast();
    }
  }, [forecastDept, forecastKpi, activeTab]);

  const handleSummarize = async () => {
    if (!summaryInput.trim()) return;
    try {
      setSummaryLoading(true);
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: summaryInput })
      });
      const data = await response.json();
      if (response.ok) {
        setSummaryResult(data.summary);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setSummaryLoading(false);
    }
  };

  const handleSentiment = async () => {
    if (!sentimentInput.trim()) return;
    try {
      setSentimentLoading(true);
      const response = await fetch('/api/analyze-sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentimentInput })
      });
      const data = await response.json();
      if (response.ok) {
        setSentimentResult(data);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setSentimentLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setUploadMessage({ text: 'Please select a CSV file first.', type: 'error' });
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setUploading(true);
      setUploadMessage({ text: '', type: '' });
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (response.ok) {
        setUploadMessage({ text: 'File uploaded and AI engine re-trained successfully!', type: 'success' });
        if (!isConfigured) {
          fetchMetrics([], '', '');
        }
      } else {
        setUploadMessage({ text: data.error || 'Upload failed.', type: 'error' });
      }
    } catch (error) {
      setUploadMessage({ text: 'An error occurred during file upload.', type: 'error' });
    } finally {
      setUploading(false);
    }
  };

  const calculateAverages = () => {
    const filteredRecords = records.filter(r => {
      if (selectedDepts.length > 0 && !selectedDepts.includes(r.department)) return false;
      if (dateRange.start && r.date < dateRange.start) return false;
      if (dateRange.end && r.date > dateRange.end) return false;
      return true;
    });

    if (filteredRecords.length === 0) return { attrition: 0, engagement: 0, delay: 0, attritionChange: 0, engagementChange: 0 };
    const total = filteredRecords.reduce((acc, curr) => {
      acc.attrition += curr.attrition_rate;
      acc.engagement += curr.engagement_score;
      acc.delay += curr.project_delay_index;
      acc.attritionChange += curr.attrition_change;
      acc.engagementChange += curr.engagement_change;
      return acc;
    }, { attrition: 0, engagement: 0, delay: 0, attritionChange: 0, engagementChange: 0 });

    const len = filteredRecords.length;
    return {
      attrition: (total.attrition / len * 100).toFixed(1),
      engagement: (total.engagement / len * 100).toFixed(1),
      delay: (total.delay / len * 100).toFixed(1),
      attritionChange: (total.attritionChange / len * 100).toFixed(2),
      engagementChange: (total.engagementChange / len * 100).toFixed(2)
    };
  };

  const averages = calculateAverages();

  const getFilteredRecords = () => {
    return records.filter(r => {
      if (selectedDepts.length > 0 && !selectedDepts.includes(r.department)) return false;
      if (dateRange.start && r.date < dateRange.start) return false;
      if (dateRange.end && r.date > dateRange.end) return false;
      return true;
    });
  };

  const getSentimentChartData = () => {
    const deptSentiments = {};
    const filtered = getFilteredRecords();
    
    allDepartments.forEach(dept => {
      deptSentiments[dept] = { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 };
    });

    filtered.forEach(r => {
      if (deptSentiments[r.department]) {
        const label = r.sentiment_label ? r.sentiment_label.toUpperCase() : 'NEUTRAL';
        deptSentiments[r.department][label] = (deptSentiments[r.department][label] || 0) + 1;
      }
    });

    const labels = Object.keys(deptSentiments);
    const positiveData = labels.map(l => deptSentiments[l].POSITIVE);
    const negativeData = labels.map(l => deptSentiments[l].NEGATIVE);
    const neutralData = labels.map(l => deptSentiments[l].NEUTRAL);

    return {
      labels,
      datasets: [
        {
          label: 'Positive Sentiment',
          data: positiveData,
          backgroundColor: 'rgba(52, 211, 153, 0.7)',
          borderColor: 'rgb(52, 211, 153)',
          borderWidth: 1,
        },
        {
          label: 'Neutral Sentiment',
          data: neutralData,
          backgroundColor: 'rgba(156, 163, 175, 0.7)',
          borderColor: 'rgb(156, 163, 175)',
          borderWidth: 1,
        },
        {
          label: 'Negative Sentiment',
          data: negativeData,
          backgroundColor: 'rgba(239, 68, 68, 0.7)',
          borderColor: 'rgb(239, 68, 68)',
          borderWidth: 1,
        }
      ]
    };
  };

  const getForecastChartData = () => {
    const labels = forecastList.map(item => item.date);
    const actuals = forecastList.map(item => item.actual);
    const forecasts = forecastList.map(item => item.forecast);

    return {
      labels,
      datasets: [
        {
          label: 'Actual KPI',
          data: actuals,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          pointBackgroundColor: '#6366f1',
          fill: true,
          tension: 0.3,
          borderWidth: 3
        },
        {
          label: 'Forecast Trend',
          data: forecasts,
          borderColor: '#f59e0b',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          pointBackgroundColor: '#f59e0b',
          tension: 0.1,
          borderWidth: 2
        }
      ]
    };
  };

  if (showSplash) {
    return (
      <div className={`splash-screen ${splashFade ? 'fade-out-splash' : ''}`}>
        <div className="splash-content">
          <div className="pulse-brain">🧠</div>
          <h1 className="splash-title">SmartCorp AI</h1>
          <p className="splash-subtitle">Corporate Insights Platform</p>
          <div className="splash-progress-bar">
            <div className="splash-progress-fill"></div>
          </div>
          <p className="splash-message">{splashMessage}</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="auth-container">
        <div className="auth-card glow-blue fade-in">
          <div className="auth-header">
            <span className="auth-logo">🧠</span>
            <h1>SmartCorp AI</h1>
            <p>{isRegistering ? 'Register Corporate Account' : 'Executive Portal Login'}</p>
          </div>
          
          <form onSubmit={isRegistering ? handleRegister : handleLogin} className="auth-form">
            <div className="input-box">
              <label>Email Address</label>
              <input 
                type="email" 
                required 
                value={email} 
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@company.com"
              />
            </div>
            
            <div className="input-box">
              <label>Password</label>
              <input 
                type="password" 
                required 
                value={password} 
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>

            <button type="submit" className="auth-submit-btn">
              {isRegistering ? 'Create Account' : 'Authenticate Credentials'}
            </button>
          </form>

          {authError && <div className="message-banner error">{authError}</div>}

          <div className="auth-toggle">
            {isRegistering ? (
              <p>Already have an account? <span onClick={() => { setIsRegistering(false); setAuthError(''); }}>Log In</span></p>
            ) : (
              <p>New executive user? <span onClick={() => { setIsRegistering(true); setAuthError(''); }}>Register</span></p>
            )}
          </div>
        </div>
      </div>
    );
  }

  const filteredRecords = getFilteredRecords();

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo-icon">🧠</span>
          <h2>SmartCorp AI</h2>
        </div>
        <p className="tagline">Corporate Insights Platform</p>

        <hr className="divider" />

        <div className="user-profile">
          <p className="user-email">{user.email}</p>
          <span className="logout-link" onClick={handleLogout}>Log Out</span>
        </div>

        {!isConfigured && (
          <div className="sandbox-badge">
            ⚡ Local Sandbox Mode
          </div>
        )}

        <hr className="divider" />

        <div className="filter-group">
          <h3>Departments</h3>
          <div className="checkbox-list">
            {allDepartments.map(dept => (
              <label key={dept} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedDepts.includes(dept)}
                  onChange={() => handleDeptToggle(dept)}
                />
                <span className="checkbox-custom"></span>
                {dept}
              </label>
            ))}
          </div>
        </div>

        <div className="filter-group">
          <h3>Date Range</h3>
          <div className="date-inputs">
            <div className="input-box">
              <label>Start Date</label>
              <input
                type="date"
                value={dateRange.start}
                min={dateRange.min}
                max={dateRange.max}
                onChange={(e) => handleDateChange('start', e.target.value)}
              />
            </div>
            <div className="input-box">
              <label>End Date</label>
              <input
                type="date"
                value={dateRange.end}
                min={dateRange.min}
                max={dateRange.max}
                onChange={(e) => handleDateChange('end', e.target.value)}
              />
            </div>
          </div>
        </div>

        <button className="reset-btn" onClick={resetFilters}>Reset Filters</button>
      </aside>

      <main className="main-content">
        <header className="tabs-header">
          <div className="tabs">
            <button
              className={`tab-link ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              📊 Executive Overview
            </button>
            <button
              className={`tab-link ${activeTab === 'nlp' ? 'active' : ''}`}
              onClick={() => setActiveTab('nlp')}
            >
              🗣️ NLP Morale Hub
            </button>
            <button
              className={`tab-link ${activeTab === 'forecast' ? 'active' : ''}`}
              onClick={() => setActiveTab('forecast')}
            >
              📈 KPI Trends & Forecast
            </button>
            <button
              className={`tab-link ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              📁 Data Ingest
            </button>
          </div>
        </header>

        {loading ? (
          <div className="loader-container">
            <div className="loader"></div>
            <p>Evaluating enterprise data models...</p>
          </div>
        ) : (
          <div className="tab-pane-content">
            {activeTab === 'overview' && (
              <div className="fade-in">
                <div className="stats-row">
                  <div className="card stat-card glow-blue">
                    <span className="stat-icon">📉</span>
                    <div className="stat-info">
                      <h4>Avg Attrition Rate</h4>
                      <p className="stat-value">{averages.attrition}%</p>
                      <span className={`stat-delta ${parseFloat(averages.attritionChange) >= 0 ? 'bad' : 'good'}`}>
                        {parseFloat(averages.attritionChange) >= 0 ? '▲' : '▼'} {Math.abs(averages.attritionChange)}% MoM
                      </span>
                    </div>
                  </div>

                  <div className="card stat-card glow-green">
                    <span className="stat-icon">🤝</span>
                    <div className="stat-info">
                      <h4>Avg Engagement Score</h4>
                      <p className="stat-value">{averages.engagement}%</p>
                      <span className={`stat-delta ${parseFloat(averages.engagementChange) >= 0 ? 'good' : 'bad'}`}>
                        {parseFloat(averages.engagementChange) >= 0 ? '▲' : '▼'} {Math.abs(averages.engagementChange)}% MoM
                      </span>
                    </div>
                  </div>

                  <div className="card stat-card glow-purple">
                    <span className="stat-icon">🕒</span>
                    <div className="stat-info">
                      <h4>Avg Project Delay Index</h4>
                      <p className="stat-value">{averages.delay}%</p>
                      <span className="stat-delta neutral">Steady</span>
                    </div>
                  </div>
                </div>

                <div className="card section-card">
                  <h2>Predicted Risk Audit (AI Classifier)</h2>
                  <p className="section-description">
                    Utilizes a Decision Tree Classifier trained on global corporate data to map threat indexes by department.
                  </p>
                  
                  {filteredRecords.length === 0 ? (
                    <p className="no-data">No records match the current department or date boundaries.</p>
                  ) : (
                    <div className="table-responsive">
                      <table className="risk-table">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Department</th>
                            <th>Meeting Summary</th>
                            <th>Risk Assessment</th>
                            <th>Morale Label</th>
                            <th>Raw Employee Feedback</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredRecords.map((r, idx) => (
                            <tr key={idx}>
                              <td>{r.date}</td>
                              <td className="bold">{r.department}</td>
                              <td className="text-summary" title={r.meeting_transcript}>
                                {r.meeting_summary || r.meeting_transcript}
                              </td>
                              <td>
                                <span className={`badge risk-${r.predicted_risk.toLowerCase()}`}>
                                  {r.predicted_risk}
                                </span>
                              </td>
                              <td>
                                <span className={`badge sentiment-${r.sentiment_label.toLowerCase()}`}>
                                  {r.sentiment_label}
                                </span>
                              </td>
                              <td className="text-feedback" title={r.employee_feedback}>
                                "{r.employee_feedback}"
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'nlp' && (
              <div className="fade-in grid-2-columns">
                <div className="left-column">
                  <div className="card chart-card">
                    <h2>Sentiment Distribution</h2>
                    <p className="section-description">Breakdown of positive, neutral, and negative expressions parsed by DistilBERT.</p>
                    <div className="chart-container">
                      <Bar 
                        data={getSentimentChartData()}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            x: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                            y: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                          },
                          plugins: {
                            legend: { labels: { color: '#cbd5e1' } }
                          }
                        }}
                      />
                    </div>
                  </div>

                  <div className="card nlp-card">
                    <h2>Live Sentiment Evaluator</h2>
                    <p className="section-description">Test individual employee feedback strings in our fine-tuned classifier.</p>
                    <textarea
                      placeholder="e.g. Workload has been heavily intense lately and we are struggling to meet target dates..."
                      value={sentimentInput}
                      onChange={(e) => setSentimentInput(e.target.value)}
                      rows="3"
                    ></textarea>
                    <button 
                      onClick={handleSentiment} 
                      disabled={sentimentLoading || !sentimentInput.trim()}
                      className="nlp-btn"
                    >
                      {sentimentLoading ? 'Classifying morale...' : 'Evaluate Morale'}
                    </button>
                    {sentimentResult && (
                      <div className="nlp-result fade-in">
                        <h4>Analysis Result:</h4>
                        <div className="result-row">
                          <span className={`badge sentiment-${sentimentResult.label.toLowerCase()}`}>
                            {sentimentResult.label}
                          </span>
                          <span className="result-score">Confidence Score: {(sentimentResult.score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="right-column">
                  <div className="card nlp-card summarizer-card">
                    <h2>BART Meeting Summarizer</h2>
                    <p className="section-description">Condense multi-page raw transcripts into high-level action summaries.</p>
                    <textarea
                      placeholder="Paste meeting transcript here..."
                      value={summaryInput}
                      onChange={(e) => setSummaryInput(e.target.value)}
                      rows="10"
                    ></textarea>
                    <button 
                      onClick={handleSummarize} 
                      disabled={summaryLoading || !summaryInput.trim()}
                      className="nlp-btn glow"
                    >
                      {summaryLoading ? 'Processing CNN-BART model...' : 'Generate Executive Summary'}
                    </button>
                    {summaryResult && (
                      <div className="nlp-result summary-box fade-in">
                        <h4>Executive Summary:</h4>
                        <p>{summaryResult}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'forecast' && (
              <div className="fade-in">
                <div className="card section-card forecast-controls-card">
                  <h2>Linear Regression KPI Forecasting</h2>
                  <p className="section-description">
                    Fit a regression trend line on historical data to predict future paths of departmental performance.
                  </p>
                  
                  <div className="forecast-controls">
                    <div className="control-item">
                      <label>Department Target</label>
                      <select value={forecastDept} onChange={(e) => setForecastDept(e.target.value)}>
                        {allDepartments.map(d => (
                          <option key={d} value={d}>{d}</option>
                        ))}
                      </select>
                    </div>

                    <div className="control-item">
                      <label>Metric KPI</label>
                      <select value={forecastKpi} onChange={(e) => setForecastKpi(e.target.value)}>
                        <option value="attrition_rate">Attrition Rate</option>
                        <option value="engagement_score">Engagement Score</option>
                        <option value="project_delay_index">Project Delay Index</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="card chart-card full-width-chart">
                  <div className="chart-title-row">
                    <h2>Trend Overlay: {forecastDept} ({forecastKpi.replace('_', ' ')})</h2>
                  </div>
                  {forecastLoading ? (
                    <div className="chart-loader">
                      <div className="spinner"></div>
                      <p>Calculating regressions...</p>
                    </div>
                  ) : forecastList.length === 0 ? (
                    <p className="no-data">No historical metrics to model.</p>
                  ) : (
                    <div className="chart-container-large">
                      <Line 
                        data={getForecastChartData()}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                          },
                          plugins: {
                            legend: { labels: { color: '#cbd5e1' } }
                          }
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'upload' && (
              <div className="fade-in upload-container">
                <div className="card upload-card glow-blue">
                  <h2>Ingest Survey Dataset</h2>
                  <p className="section-description">
                    Upload your department logs in CSV format to trigger retraining of Decision Tree risk predictions.
                  </p>
                  
                  <form onSubmit={handleFileUpload} className="upload-form">
                    <div className="file-dropzone">
                      <input 
                        type="file" 
                        accept=".csv" 
                        id="csv-file-input"
                        onChange={(e) => setSelectedFile(e.target.files[0])}
                      />
                      <label htmlFor="csv-file-input" className="file-label">
                        <span className="upload-cloud">📁</span>
                        {selectedFile ? (
                          <span className="file-name-display">{selectedFile.name}</span>
                        ) : (
                          <span>Drag and drop your company CSV here or <span className="highlight-text">browse files</span></span>
                        )}
                      </label>
                    </div>

                    <button 
                      type="submit" 
                      className="upload-submit-btn"
                      disabled={uploading || !selectedFile}
                    >
                      {uploading ? 'Parsing CSV & Re-training Decision Trees...' : 'Ingest and Update Models'}
                    </button>
                  </form>

                  {uploadMessage.text && (
                    <div className={`message-banner ${uploadMessage.type} fade-in`}>
                      {uploadMessage.text}
                    </div>
                  )}

                  <div className="csv-specs">
                    <h3>Required Column Format:</h3>
                    <ul>
                      <li><code>date</code> (YYYY-MM-DD format)</li>
                      <li><code>department</code> (e.g. Engineering, Sales, HR, Marketing)</li>
                      <li><code>meeting_transcript</code> (raw transcription string)</li>
                      <li><code>employee_feedback</code> (anonymous employee survey text)</li>
                      <li><code>attrition_rate</code> (float between 0.00 and 1.00)</li>
                      <li><code>engagement_score</code> (float between 0.00 and 1.00)</li>
                      <li><code>project_delay_index</code> (float between 0.00 and 1.00)</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
