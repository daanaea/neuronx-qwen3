#!/usr/bin/env python3
"""
Auto-stop utility for AWS EC2 instances.
Automatically stops the instance after a specified time period.

Requires:
- IAM role attached to EC2 instance with ec2:StopInstances permission
- boto3 library
- requests library
"""

import logging
import os
import threading
import time
import boto3
import requests

logger = logging.getLogger(__name__)


def auto_stop(delay_seconds=60, region="us-east-1", fallback_to_local=True):
    """
    Automatically stops this EC2 instance after `delay_seconds` seconds.
    Works via AWS API for clean shutdown with proper state preservation.

    Args:
        delay_seconds (int): Time to wait before stopping the instance (default: 60)
        region (str): AWS region where the instance is running (default: us-east-1)
        fallback_to_local (bool): If True, falls back to local shutdown if AWS API fails

    Requires IAM role with ec2:StopInstances permission attached to the instance.

    Example IAM Policy:
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": "ec2:StopInstances",
          "Resource": "*"
        }
      ]
    }
    """

    def stop_instance():
        logger.info(f"[Auto-Stop] Timer started. Instance will stop in {delay_seconds} seconds ({delay_seconds/60:.1f} minutes)...")
        time.sleep(delay_seconds)

        try:
            # Get the instance ID from EC2 metadata service
            logger.info("[Auto-Stop] Retrieving instance ID from EC2 metadata...")
            instance_id = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id",
                timeout=2
            ).text.strip()

            logger.info(f"[Auto-Stop] Instance ID: {instance_id}")
            logger.info(f"[Auto-Stop] Stopping instance {instance_id} via AWS API...")

            # Initialize boto3 EC2 client
            ec2 = boto3.client("ec2", region_name=region)

            # Stop the instance
            response = ec2.stop_instances(InstanceIds=[instance_id])
            logger.info(f"[Auto-Stop] Stop signal sent successfully. Response: {response}")
            logger.info("[Auto-Stop] Instance will shut down shortly. Goodbye!")

        except requests.exceptions.RequestException as e:
            logger.error(f"[Auto-Stop] Failed to retrieve instance ID from metadata service: {e}")
            logger.warning("[Auto-Stop] This may not be running on an EC2 instance or metadata service is unavailable.")
            if fallback_to_local:
                logger.info("[Auto-Stop] Attempting local shutdown as fallback...")
                os.system("sudo shutdown -h now")

        except Exception as e:
            logger.error(f"[Auto-Stop] Failed to stop via AWS API: {e}")
            logger.error(f"[Auto-Stop] Error type: {type(e).__name__}")

            if fallback_to_local:
                logger.info("[Auto-Stop] Falling back to local shutdown command...")
                try:
                    os.system("sudo shutdown -h now")
                except Exception as local_err:
                    logger.error(f"[Auto-Stop] Local shutdown also failed: {local_err}")
            else:
                logger.error("[Auto-Stop] Fallback to local shutdown is disabled. Instance will NOT be stopped.")

    # Run in background thread so main code can continue
    stop_thread = threading.Thread(target=stop_instance, daemon=True)
    stop_thread.start()
    logger.info(f"[Auto-Stop] Background shutdown timer started (delay: {delay_seconds}s)")

    return stop_thread


def auto_stop_on_completion(func):
    """
    Decorator to automatically stop the instance after a function completes.

    Usage:
        @auto_stop_on_completion
        def train_model():
            # Your training code here
            pass
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            logger.info("[Auto-Stop] Function completed successfully. Stopping instance in 30 seconds...")
            auto_stop(delay_seconds=30)
            return result
        except Exception as e:
            logger.error(f"[Auto-Stop] Function failed with error: {e}")
            logger.info("[Auto-Stop] Stopping instance in 30 seconds due to error...")
            auto_stop(delay_seconds=30)
            raise
    return wrapper


if __name__ == "__main__":
    # Test the auto-stop functionality
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing auto-stop functionality...")
    print("This will attempt to stop the instance in 10 seconds.")
    print("Press Ctrl+C to cancel.")

    try:
        auto_stop(delay_seconds=10, region="us-east-1")
        # Keep main thread alive to see the auto-stop in action
        time.sleep(15)
    except KeyboardInterrupt:
        print("\nAuto-stop cancelled by user.")
