import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Switch, Link } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const App = () => {
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
            <Dashboard patientId={1} /> {/* For demo purposes, we're using a fixed patient ID */}
          </Route>
          {/* Add more routes here as needed */}
        </Switch>
      </div>
    </Router>
  );
};

export default App;
