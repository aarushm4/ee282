export const TreatmentRecommendations = ({ recommendations }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Treatment Recommendations</h2>
    <ul>
      {Object.entries(recommendations).map(([treatment, score], index) => (
        <li key={index} className='mb-2'>
          <span className='font-semibold'>{treatment}:</span> {score.toFixed(2)}
        </li>
      ))}
    </ul>
  </div>
);