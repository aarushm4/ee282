import React, { useEffect, useState } from 'react';
import { AppRouter } from './Router';

const randomIntFromInterval = (min, max) => {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
};

const App = () => {
  const [patientId, setPatientId] = useState(-1);
  useEffect(() => {
    setPatientId(randomIntFromInterval(1, 1000));
  }, [])
  
  return (   
    <AppRouter patientId={patientId} />
  );
};

export default App;
