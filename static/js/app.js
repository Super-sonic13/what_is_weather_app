async function predict(model) {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const predictionsDiv = document.getElementById('predictions');

    const formData = new FormData(form);
    formData.append('model', model);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        predictionsDiv.innerHTML = `
            <h3 class="mt-4">Predictions:</h3>
            <p>CNN Prediction: ${result.cnn_prediction}</p>
            <p>RandomForest Prediction: ${result.rf_prediction}</p>
        `;

        fileInput.value = ''; // Clear file input
    } catch (error) {
        console.error('Error during prediction:', error);
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const cnnButton = document.getElementById('cnn-button');
    const rfButton = document.getElementById('rf-button');

    cnnButton.addEventListener('click', () => predict('cnn'));
    rfButton.addEventListener('click', () => predict('rf'));
});
