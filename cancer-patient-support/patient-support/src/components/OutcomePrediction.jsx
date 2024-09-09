export const OutcomePrediction = ({ prediction }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Outcome Prediction</h2>
    <div className='flex items-center'>
      <div className='w-24 h-24 rounded-full bg-blue-500 flex items-center justify-center text-white text-2xl font-bold'>{Math.round(prediction.improvement_probability * 100)}%</div>
      <p className='ml-4'>Probability of improvement</p>
    </div>
  </div>
);
