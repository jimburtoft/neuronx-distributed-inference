#!/bin/bash
# Task 008: Launch trn2.3xlarge for Qwen2.5-Omni M-RoPE validation
# Run this AFTER the capacity block cr-03a382fd3d7a47da5 becomes active (11:30 UTC Apr 11)

set -e

REGION="sa-east-1"
CR_ID="cr-03a382fd3d7a47da5"
AMI_ID="ami-0b0749742fb2391dc"  # SDK 2.29 DLAMI
INSTANCE_TYPE="trn2.3xlarge"
KEY_NAME="SaoPaulo"
SG_ID="sg-081c41fb27efb1555"
SUBNET_ID="subnet-03257e446d18ec227"

echo "Checking capacity block status..."
STATE=$(aws ec2 describe-capacity-reservations --capacity-reservation-ids $CR_ID --region $REGION --query 'CapacityReservations[0].State' --output text)
echo "Capacity block state: $STATE"

if [ "$STATE" != "active" ]; then
    echo "ERROR: Capacity block is not active yet (state: $STATE). Wait until 11:30 UTC."
    exit 1
fi

echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --region $REGION \
    --instance-market-options '{"MarketType": "capacity-block"}' \
    --capacity-reservation-specification "{\"CapacityReservationTarget\": {\"CapacityReservationId\": \"$CR_ID\"}}" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=OC-sweeper-omni-mrope}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance running at: $IP"
echo ""
echo "SSH: ssh -i ~/.ssh/SaoPaulo.pem ubuntu@$IP"
