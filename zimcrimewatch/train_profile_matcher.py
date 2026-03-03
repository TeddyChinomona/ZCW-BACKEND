"""
Management command: train_profile_matcher
Usage: python manage.py train_profile_matcher
       python manage.py train_profile_matcher --min-samples 5

Trains (or retrains) the Random Forest crime profile matching model
using all CrimeIncidents that have a non-empty serial_group_label.
"""
from django.core.management.base import BaseCommand
from .models import CrimeIncident
from .ml_utils import ProfileMatcher


class Command(BaseCommand):
    help = "Train the Random Forest crime profile matching model."

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-samples",
            type=int,
            default=10,
            help="Minimum number of labelled incidents required to train (default: 10).",
        )

    def handle(self, *args, **options):
        min_samples = options["min_samples"]
        qs = CrimeIncident.objects.exclude(serial_group_label="").select_related("crime_type")
        count = qs.count()

        self.stdout.write(f"Found {count} labelled incidents.")
        if count < min_samples:
            self.stderr.write(
                self.style.ERROR(
                    f"Need at least {min_samples} labelled incidents. "
                    f"Add 'serial_group_label' values to incidents in the admin or via the API."
                )
            )
            return

        self.stdout.write("Training Random Forest model...")
        matcher = ProfileMatcher()
        metrics = matcher.train(qs)

        if "error" in metrics:
            self.stderr.write(self.style.ERROR(metrics["error"]))
            return

        self.stdout.write(self.style.SUCCESS(
            f"✓ Model trained successfully!\n"
            f"  Samples:       {metrics['n_samples']}\n"
            f"  Classes:       {metrics['n_classes']}\n"
            f"  CV Accuracy:   {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}\n"
            f"  Model saved to ml_models/profile_matcher.pkl"
        ))
