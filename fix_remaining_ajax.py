#!/usr/bin/env python3
"""
Script to apply AJAX fixes to remaining prediction pages
"""

import os
import re

def fix_template_ajax(template_path, form_id, submit_text, endpoint):
    """Apply AJAX fixes to a template"""
    print(f"🔧 Fixing {template_path}...")
    
    # Read the template
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add results container and debug sections after existing results
    results_section = r'(<!-- Results -->.*?{% endif %})'
    replacement = r'''\1

            <!-- Results Container for AJAX -->
            <div id="resultsContainer" style="display: none;">
                <!-- Results will be displayed here via JavaScript -->
            </div>

            <!-- Error Display -->
            {% if error %}
            <div class="alert alert-danger mt-4">
                <h5><i class="bi bi-exclamation-triangle"></i> Error</h5>
                <p class="mb-0">{{ error }}</p>
            </div>
            {% endif %}

            <!-- Debug Info -->
            {% if debug_info %}
            <div class="alert alert-info mt-4">
                <h5><i class="bi bi-bug"></i> Debug Info</h5>
                <p class="mb-0">{{ debug_info }}</p>
            </div>
            {% endif %}

            <!-- Form Debug -->
            <div class="card mt-4">
                <div class="card-header bg-warning text-dark">
                    <h5 class="mb-0">
                        <i class="bi bi-bug"></i> Form Debug (This will help us see what's happening)
                    </h5>
                </div>
                <div class="card-body">
                    <div id="formDebug">
                        <p>Form ready. Upload an image and click "''' + submit_text + '''" to see debug info.</p>
                    </div>
                </div>
            </div>'''
    
    content = re.sub(results_section, replacement, content, flags=re.DOTALL)
    
    # Update JavaScript to use AJAX
    js_pattern = r'(// Form submission.*?});)'
    js_replacement = f'''// Form submission
    form.addEventListener('submit', function(e) {{
        e.preventDefault(); // PREVENT DEFAULT FORM SUBMISSION
        console.log('{form_id} form submission started');
        
        // Update debug info
        document.getElementById('formDebug').innerHTML = '<p class="text-info">🔄 Form submission started...</p>';
        
        if (!imageInput.files[0]) {{
            alert('Please select an image file first.');
            document.getElementById('formDebug').innerHTML = '<p class="text-danger">❌ No image file selected</p>';
            return;
        }}
        
        console.log('Image file selected:', imageInput.files[0].name);
        document.getElementById('formDebug').innerHTML = `
            <p class="text-success">✅ Image file selected: ${{imageInput.files[0].name}}</p>
            <p class="text-info">📁 File size: ${{(imageInput.files[0].size / 1024).toFixed(1)}} KB</p>
            <p class="text-info">🔄 Submitting form via AJAX...</p>
        `;
        
        // Show loading state
        submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> {submit_text}...';
        submitBtn.disabled = true;
        
        // Submit form via AJAX
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);
        
        console.log('FormData contents:');
        for (let [key, value] of formData.entries()) {{
            console.log(key, value);
        }}
        
        fetch('{endpoint}', {{
            method: 'POST',
            body: formData
        }})
        .then(response => {{
            console.log('Response received:', response.status);
            document.getElementById('formDebug').innerHTML += '<p class="text-info">📡 Response received from server</p>';
            return response.text();
        }})
        .then(html => {{
            console.log('Response HTML received');
            document.getElementById('formDebug').innerHTML += '<p class="text-success">✅ Server response received</p>';
            
            // Parse the response and extract debug info
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            
            // Check for debug info
            const debugInfo = doc.querySelector('.alert-info');
            if (debugInfo) {{
                document.getElementById('formDebug').innerHTML += `<p class="text-info">🔍 ${{debugInfo.textContent.trim()}}</p>`;
            }}
            
            // Check for results and display them
            const resultCard = doc.querySelector('.result-card');
            if (resultCard) {{
                document.getElementById('formDebug').innerHTML += '<p class="text-success">🎉 Results found in response!</p>';
                
                // Display the results in the main results section
                const resultsContainer = document.querySelector('#resultsContainer');
                if (resultsContainer) {{
                    resultsContainer.innerHTML = resultCard.innerHTML;
                    resultsContainer.style.display = 'block';
                }}
            }}
            
            // Check for errors
            const errorAlert = doc.querySelector('.alert-danger');
            if (errorAlert) {{
                document.getElementById('formDebug').innerHTML += `<p class="text-danger">❌ Error: ${{errorAlert.textContent.trim()}}</p>`;
            }}
            
            // Reset button
            submitBtn.innerHTML = '<i class="bi bi-cpu"></i> {submit_text}';
            submitBtn.disabled = false;
        }})
        .catch(error => {{
            console.error('Error:', error);
            document.getElementById('formDebug').innerHTML += `<p class="text-danger">❌ AJAX Error: ${{error}}</p>`;
            
            // Reset button
            submitBtn.innerHTML = '<i class="bi bi-cpu"></i> {submit_text}';
            submitBtn.disabled = false;
        }});
        
        console.log('Form submitted via AJAX, waiting for response...');
    }});'''
    
    content = re.sub(js_pattern, js_replacement, content, flags=re.DOTALL)
    
    # Fix the change file button
    button_pattern = r'onclick="document\.getElementById\(\'imageInput\'\)\.click\(\)"'
    button_replacement = 'id="changeFileBtn"'
    content = re.sub(button_pattern, button_replacement, content)
    
    # Add event listener for change file button
    change_file_js = '''
            // Add event listener to the new button
            document.getElementById('changeFileBtn').addEventListener('click', function() {
                document.getElementById('imageInput').click();
            });'''
    
    content = content.replace('            `;', '            `;' + change_file_js)
    
    # Write the updated template
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed {template_path}")

def main():
    """Apply AJAX fixes to remaining templates"""
    print("🔧 APPLYING AJAX FIXES TO REMAINING PAGES")
    print("=" * 60)
    
    templates = [
        ('livestock_ai_webapp/templates/predictions/hoofed_animals.html', 'hoofedAnimalsForm', 'Analyze Hoofed Animals', '/hoofed-animals/'),
        ('livestock_ai_webapp/templates/predictions/disease_detection.html', 'diseaseForm', 'Detect Disease', '/disease/')
    ]
    
    for template_path, form_id, submit_text, endpoint in templates:
        if os.path.exists(template_path):
            fix_template_ajax(template_path, form_id, submit_text, endpoint)
        else:
            print(f"❌ Template not found: {template_path}")
    
    print("\n🎉 All AJAX fixes applied!")

if __name__ == "__main__":
    main()
