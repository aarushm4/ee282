import React from 'react';
import { BrowserRouter as Router, Route, Switch, Link } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Dashboard } from './components/Dashboard';

const randomIntFromInterval = (min, max) => {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
};

const App = () => {
  const patientid = randomIntFromInterval(1, 1000);
  return (
    <Router>
      <div className='min-h-screen bg-gray-100'>
        <nav className='bg-blue-600 text-white p-4'>
          <div className='container mx-auto'>
            <Link to='/' className='text-2xl font-bold'>
              Cancer Patient Support
            </Link>
          </div>
        </nav>
        <Switch>
          <Route exact path='/'>
            <Dashboard patientId={patientid} /> {/* For demo purposes, we're using a fixed patient ID */}
          </Route>
          {/* Add more routes here as needed */}
        </Switch>
      </div>
    </Router>
  );
};

export default App;
