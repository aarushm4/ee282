export const ClinicalTrials = ({ trials }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Relevant Clinical Trials</h2>
    <ul className='list-disc pl-5'>
      {trials.map((trial, index) => (
        <li key={index} className='mb-2'>
          <a href={trial.link} target='_blank' rel='noopener noreferrer' className='text-blue-500 hover:underline'>
            {trial.title}
          </a>
        </li>
      ))}
    </ul>
  </div>
);