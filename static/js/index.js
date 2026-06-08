window.HELP_IMPROVE_VIDEOJS = false;

document.addEventListener("DOMContentLoaded", function () {
  const emailButton = document.getElementById("dataset-email-btn");

  if (emailButton) {
    emailButton.addEventListener("click", function (event) {
      // Prevents the page from jumping up due to href="#"
      event.preventDefault(); 

      const email = "tim.walter@tum.de";
      const subject = "RAM: Dataset inquiry";
      
      // Write your email template normally in plain text here
      const body = `Dear Tim Walter,

      My name is [Your Name], and I am a [Your Position/Role] at [Your Institution/Company]. 
      I am interested in using your RAM dataset. Could you please provide access?

Best regards,
[Your Name]`;

      // Construct the mailto link dynamically using built-in encoding
      const mailtoUrl = `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

      // Open the user's default email platform
      window.location.href = mailtoUrl;
    });
  }
});

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Copied';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Copied';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Attach to standard generic carousels across the site if any exist
    var otherCarousels = bulmaCarousel.attach('.carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000
    });
    
    bulmaSlider.attach();
    setupVideoCarouselAutoplay();

    // Custom Adaptive Visualisation Router for the Dataset Section
    function switchDatasetSlide(index) {
        // 1. Synchronize Highlight Triggers
        $('.pipeline-trigger').removeClass('is-active');
        $('.pipeline-trigger[data-slide-index="' + index + '"]').addClass('is-active');

        // 2. Cycle Slides and Adapt Layout Heights Naturally
        $('.dataset-slide').removeClass('is-active').hide();
        
        var $targetSlide = $('.dataset-slide').eq(index);
        $targetSlide.show();
        setTimeout(function() {
            $targetSlide.addClass('is-active');
        }, 15);

        // 3. Contextually manage the video stream runtime
        var labelVideo = document.getElementById('dataset-video-label');
        if (labelVideo) {
            if (parseInt(index, 10) === 2) {
                labelVideo.play().catch(function(e) {});
            } else {
                labelVideo.pause();
            }
        }
    }

    // Capture explicit layout selections from user interactions
    $('.pipeline-trigger').on('click', function() {
        var targetIndex = parseInt($(this).attr('data-slide-index'), 10);
        switchDatasetSlide(targetIndex);
    });
});