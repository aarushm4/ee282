import { BrowserRouter as Router, Route, Switch, Link } from 'react-router-dom';
import { Dashboard } from './components/Dashboard';

export const AppRouter = ({ patientId }) => {
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
          <Route exact path='/dashboard'>
            <Dashboard patientId={patientId} />
          </Route>
          {/* Add more routes here as needed */}
        </Switch>
      </div>
    </Router>
  );
};
