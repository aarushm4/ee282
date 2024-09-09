export const PatientInfo = ({ patientInfo }) => (
  <div className='bg-white shadow rounded-lg p-6'>
    <h2 className='text-xl font-semibold mb-4'>Patient Information</h2>
    <p>
      <strong>Age:</strong> {patientInfo.age}
    </p>
    <p>
      <strong>Cancer Type:</strong> {patientInfo.cancer_type}
    </p>
    <p>
      <strong>Stage:</strong> {patientInfo.stage}
    </p>
    <p>
      <strong>Current Treatment:</strong> {patientInfo.treatment}
    </p>
    <p>
      <strong>Genetic Marker:</strong> {patientInfo.genetic_marker}
    </p>
    <p>
      <strong>Previous Treatments:</strong> {patientInfo.previous_treatments}
    </p>
  </div>
);